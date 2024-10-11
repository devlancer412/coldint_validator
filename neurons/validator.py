# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 const

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import huggingface_hub.constants
huggingface_hub.constants.HF_HUB_DISABLE_TELEMETRY = True
# Due to the implementation of disable_progress_bars(), this has to be the first import+call in the application relating to huggingface
from huggingface_hub.utils import disable_progress_bars
disable_progress_bars()

import sys
import signal

# Install a signal handler for clean shutdown; class Validator will install its
# own shutdown handler.
if __name__ == "__main__":
    def early_shutdown(signum,stackframe):
        print('SIGINT caught, exiting',file=sys.stderr)
        sys.exit(-1)

    signal.signal(signal.SIGINT, early_shutdown)

import copy
import datetime as dt
import functools
import os
import orjson
import math
import pickle
import psutil
import time
import torch
import random
import asyncio
import pathlib
import numpy as np
import requests
import transformers
from packaging.version import Version

import wandb
import constants
import dataset
import validation
from model import model_utils, competitions
from model.data import ModelId, ModelMetadata
from model.model_updater import ModelUpdater
from model.storage.disk.disk_model_store import DiskModelStore
from model.storage.hugging_face.hugging_face_model_store import HuggingFaceModelStore
from model.storage.disk import utils as disk_utils
from neurons import config
import traceback
import threading
import multiprocessing
from rich.table import Table
from rich.console import Console

import bittensor as bt
from utilities import utils, btlite
from utilities.perf_monitor import PerfMonitor
from utilities.mathutils import *

TRANSFORMERS_VERSION_MIN     = "4.41.2"
TRANSFORMERS_VERSION_OPTIMAL = "4.44.0"

PRIO_POOL       = 100
PRIO_TOP_MODEL  = 80
PRIO_INJECT     = 60
PRIO_NEW_MODEL  = 40
PRIO_REVISIT    = 20

os.environ["TOKENIZERS_PARALLELISM"] = "true"

class Container:
    '''Empty container object'''
    pass

class ModelIssue(Exception):
    '''
    Exception class to signal issues with models preventing evaluation.
    '''
    pass

class Validator:
    STATE_FILENAME = "validator_state.json"
    BENCHMARK_FILENAME = "benchmark.json"

    def state_path(self) -> str:
        return os.path.join(self.config.model_dir, "vali-state")

    def load_state(self):
        # Updated by model updater thread, used by evaluation thread
        self.hall_of_fame = {}
        self.competitions = {}

        # Last known hotkey metadata
        self.hk_metadata = {}

        # Competition state, only used by evaluation thread
        self.cstate = {}

        state = {}
        state_fn = os.path.join(self.state_path(), Validator.STATE_FILENAME)
        if os.path.exists(state_fn):
            try:
                with open(state_fn, 'rb') as f:
                    state = orjson.loads(f.read())
            except Exception as e:
                bt.logging.info(f"Invalid state file: {e}")

        self.force_eval_until_uid_last = None
        self.last_weights_set = time.time()
        with self.state_lock:
            if 'version' in state and state['version']//10 == constants.__spec_version__//10:
                # major.minor of version has not changed
                self.hall_of_fame = state.pop("hall_of_fame", {})
                self.competitions = state.pop("competitions", {})
                self.hk_metadata  = state.pop("hk_metadata", {})
                self.hk_metadata = {hotkey: ModelMetadata(**info) for hotkey, info in self.hk_metadata.items()}
                self.cstate = state.pop('cstate', {})
                for cname, cinfo in self.cstate.items():
                    cinfo['uids_pending'] = {int(uid): prio for uid, prio in cinfo.get('uids_pending', {}).items()}
                    cinfo['uids_weight'] = {int(uid): w for uid, w in cinfo.get('uids_weight', {}).items()}
                bt.logging.info("State loaded successfully")
                self.force_eval_until_uid_last = state.pop("force_eval_until_uid_last", None)
                self.last_weights_set = state.pop("last_weights_set", time.time())
            else:
                bt.logging.info("State version incompatible, starting with clean state")

        bt.logging.debug(str(self.cstate))

    def save_state(self):
        state_dir = self.state_path()
        os.makedirs(state_dir, exist_ok=True)
        with self.state_lock:
            state = {
                'version': constants.__spec_version__,
                'competitions': self.competitions,
                'hall_of_fame': self.hall_of_fame,
                'hk_metadata': {hotkey: meta.dict() for hotkey, meta in self.hk_metadata.items() if meta is not None},
                'cstate': self.cstate,
                'force_eval_until_uid_last': self.force_eval_until_uid_last,
                'last_weights_set': self.last_weights_set,
            }
        try:
            with open(os.path.join(self.state_path(), Validator.STATE_FILENAME), 'wb') as f:
                f.write(orjson.dumps(state, option=orjson.OPT_INDENT_2|orjson.OPT_NON_STR_KEYS|orjson.OPT_SERIALIZE_NUMPY))
        except Exception as e:
            bt.logging.warning(f"Failed to save state: {e}")

    def __init__(self):
        self.config = config.validator_config()
        bt.logging(config=self.config)
        if self.config.logging.debug:
            bt.logging.set_debug(True)
        if self.config.logging.trace:
            bt.logging.set_trace(True)

        bt.logging.info(f"Starting validator with config: {self.config}")

        # === Bittensor objects ====
        self.wallet = bt.wallet(config=self.config)
        self.subtensor = btlite.get_subtensor(config=self.config)
        self.dendrite = bt.dendrite(wallet=self.wallet)
        self.metagraph = self.get_metagraph()
        torch.backends.cudnn.benchmark = True

        # Dont check registration status if offline.
        self.uid = None
        if not self.config.offline:
            self.uid = utils.assert_registered(self.wallet, self.metagraph)

        # Dont log to wandb if offline.
        self.wandb_run = None
        if self.config.wandb.on and not self.config.offline:
            self.new_wandb_run()

        # Weights and step info
        self.weights = torch.zeros(constants.SUBNET_N_UIDS)
        self.run_step_count = 0
        self.epoch_step = 0
        self.global_step = 0
        self.last_epoch = self.metagraph.block.item()
        self.last_weights_set = time.time() # Don't set weights early

        self.state_lock = threading.RLock()

        # Load or initialize internal state
        self.load_state()

        # Setup a RemoteModelStore
        self.remote_store = HuggingFaceModelStore()

        # Setup a LocalModelStore
        self.local_store = DiskModelStore(base_dir=self.config.model_dir)
        min_free_gb = constants.LIMIT_MIN_FREE_GB
        if -min_free_gb < self.config.model_store_size_gb <= 0:
            self.config.model_store_size_gb = -min_free_gb
            bt.logging.warning(f'Model store size limit set to keep {min_free_gb} GB free.')

        # Setup a model updater to download models as needed to match the latest provided miner metadata.
        bt.logging.trace("Starting ModelUpdater")
        self.model_updater = ModelUpdater(
            remote_store=self.remote_store,
            local_store=self.local_store,
            comps=self.competitions
        )

        # Create a metagraph lock to avoid cross thread access issues in the update and clean loop.
        self.metagraph_lock = threading.RLock()

        # Initialize the update thread
        self.stop_event = threading.Event()
        bt.logging.trace("Starting update thread")
        self.update_thread_ts = time.time()
        self.update_thread = threading.Thread(target=self.update_thread_func, daemon=True)
        self.update_thread.start()

    def __del__(self):
        if hasattr(self, "stop_event"):
            self.stop_event.set()
            self.update_thread.join()

    def new_wandb_run(self):
        """Creates a new wandb run to save information to."""

        # Create a unique run id for this run.
        run_id = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        name = "validator-" + str(self.uid) + "-" + run_id
        self.wandb_run = wandb.init(
            name=name,
            project=constants.WANDB_PROJECT,
            entity=constants.WANDB_ENTITY,
            config={
                "uid": self.uid,
                "hotkey": self.wallet.hotkey.ss58_address,
                "run_name": run_id,
                "version": constants.__validator_version__,
                "type": "validator",
            },
            allow_val_change=True,
        )

        bt.logging.debug(f"Started a new wandb run: {name}")

    def get_all_active_uids(self, include_pending=True):
        ret = []
        with self.state_lock:
            for cname, cinfo in self.cstate.items():
                ret.extend(cinfo['uids_pool'])
                if include_pending:
                    ret.extend(cinfo['uids_pending'].keys())
        return set(ret)

    def add_or_remove_uid_in_competition(self, uid, hotkey, prio):
        """
        Add uid to the competition pool which it participates in, remove it from others.
        """
        meta = self.hk_metadata.get(hotkey, None)
        with self.state_lock:
            for cname, cinfo in self.cstate.items():
                if meta is not None and cname == meta.id.competition:
                    bt.logging.info(f"Adding {uid} to competition {cname} with prio {prio:.02f}")
                    if uid not in cinfo['uids_pool'] and uid not in cinfo['uids_pending']:
                        cinfo['uids_pending'][uid] = prio
                else:
                    if uid in cinfo['uids_pool']:
                        cinfo['uids_pool'].remove(uid)
                        bt.logging.info(f"Removed {uid} from competition {cname} uids_pool")
                    if uid in cinfo['uids_pending']:
                        del cinfo['uids_pending'][uid]
                        bt.logging.info(f"Removed {uid} from competition {cname} uids_pending")

    def get_metagraph(self, n_retries=3):
        for i in range(n_retries):
            try:
                bt.logging.debug(f'getting metagraph from {self.subtensor} for {self.config.netuid}')
                metagraph = btlite.get_metagraph(
                    subtensor=self.subtensor,
                    netuid=self.config.netuid,
                    lite=False,
                    reconnect=False, # in case of connection issues, we re-create the subtensor
                )
                bt.logging.debug(f'got metagraph from {self.subtensor}: {metagraph}')
                if metagraph is not None:
                    return metagraph
            except Exception as e:
                bt.logging.warning(f"Failed to get metagraph {i+1}/{n_retries}: {e}\n{traceback.format_exc()}")
            bt.logging.info("Reconnecting subtensor")
            self.subtensor = btlite.get_subtensor(config=self.config)

        bt.logging.error(f"Failed to get metagraph {n_retries} times, giving up")
        return None

    def visit_uids(self, metagraph, metadata, uids, prio=0, retry_interval=0):
        '''
        Visit a list/set of uids: download model and update competition pending pools.
        If retry_interval non-zero, force re-evaluation after this interval.

        Returns the number of uids added
        '''

        if len(uids) == 0:
            return 0

        now = time.time()
        n_picked = 0
        for uid in uids:
            if uid >= len(metagraph.hotkeys):
                continue
            hotkey = metagraph.hotkeys[uid]
            hotkey_metadata = metadata.get(hotkey, None)
            model_prio = prio + random.random()

            if hotkey_metadata is None:
                self.hk_metadata[hotkey] = None
                self.add_or_remove_uid_in_competition(uid, hotkey, model_prio)
                continue

            updated = hotkey_metadata != self.hk_metadata.get(hotkey, None)
            should_retry = retry_interval != 0 and now - self.uid_last_retried_ts.get(uid,0) >= retry_interval
            if not updated and not should_retry:
                continue
            self.uid_last_retried_ts[uid] = now

            # Sync the model, if necessary.
            try:
                bt.logging.trace(f"Sync UID {uid} in PID={os.getpid()}")
                sync_result = asyncio.run(
                    self.model_updater.sync_model(hotkey, hotkey_metadata)
                )
                bt.logging.trace(f"Sync result: {sync_result}")

                if sync_result in self.model_updater.RETRY_RESULTS:
                    # By not setting self.hk_metadata[hotkey], we will retry later
                    pass
                else:
                    # Update hk_metadata, so we don't retry next loop
                    self.hk_metadata[hotkey] = hotkey_metadata

                if sync_result == self.model_updater.SYNC_RESULT_SUCCESS:
                    self.add_or_remove_uid_in_competition(uid, hotkey, model_prio)
                    n_picked += 1

            except Exception as e:
                bt.logging.error(
                    f"Failed to sync model for UID {uid}: {type(e).__name__} {e} \n {traceback.format_exc()}"
                )

        return n_picked

    def update_chain(self):
        now = time.time()
        if self.last_chain_update is not None and \
                (now - self.last_chain_update) < constants.CHAIN_UPDATE_INTERVAL:
            return

        # We would like to run chain interaction commands with a timeout, but there is no
        # straightforward method to do that using threads (as all options with a timeout
        # leave the thread running). We do not want to use subprocesses either.
        # Therefore, we run blocking in this thread. It would be nice if the bittensor library
        # timeout properties would be configurable (there are some hard-coded retries etc).
        # The main thread can monitor whether the thread performing these updates is still
        # alive, and bail out if that is not the case.
        new_metagraph = self.get_metagraph()
        if new_metagraph is None:
            bt.logging.warning(f'Failed to update metagraph')
            return
        with self.metagraph_lock:
            self.metagraph = copy.deepcopy(new_metagraph)

        self.last_chain_update = now
        bt.logging.info(f"Synced metagraph with {len(self.metagraph.neurons)} neurons, last_chain_update = {now}")

        # Fetch miner metadata
        new_metadata = {}
        for miner_uid, hotkey in enumerate(new_metagraph.hotkeys):
            try:
                mdl_metadata = btlite.get_metadata(
                    subtensor=self.subtensor,
                    netuid=self.config.netuid,
                    hotkey=hotkey,
                    reconnect=True,
                )
                if mdl_metadata is not None:
                    mdl_metadata = ModelMetadata.parse_chain_data(mdl_metadata)
                    new_metadata[hotkey] = mdl_metadata
                updated = new_metadata.get(hotkey, None) != self.hk_metadata.get(hotkey, None)
                bt.logging.debug(f"Metadata UID {miner_uid}/{hotkey}, updated={updated}, commitment: {mdl_metadata.id.format_label() if mdl_metadata else '---'}")
            except Exception as e:
                bt.logging.error(
                    f"Failed to fetch metadata for UID {miner_uid}: {type(e).__name__} {e} \n {traceback.format_exc()}"
                )
                if hotkey in self.hk_metadata:
                    bt.logging.debug(f"Using old metadata after update failure for {hotkey}")
                    new_metadata[hotkey] = self.hk_metadata[hotkey]

        bt.logging.info(f"Synced metadata; {len(new_metadata)} commitments")

        # 3-step strategy:
        # 1. Pick top miners every TOP_MODEL_RETRY_INTERVAL
        # 2. Pick new models
        # 3. Revisit old models every GEN_MODEL_RETRY_INTERVAL, slice based on timestamp
        active_uids = self.get_all_active_uids()

        # Make sure pool UIDs have metadata set
        for uid in active_uids:
            if uid >= len(new_metagraph.hotkeys):
                continue
            hotkey = new_metagraph.hotkeys[uid]
            hotkey_metadata = new_metadata.get(hotkey, None)
            self.hk_metadata[hotkey] = hotkey_metadata

        # Determine top miners according to other valis
        top_miner_uids = set(utils.list_top_miners(new_metagraph))
        new_top_uids = top_miner_uids - active_uids
        bt.logging.debug(f"Visiting top UIDs: {new_top_uids}")
        n_top_updated = self.visit_uids(
                new_metagraph,
                new_metadata,
                new_top_uids,
                prio=PRIO_TOP_MODEL,
                retry_interval=constants.TOP_MODEL_RETRY_INTERVAL,
        )

        n_uids = len(new_metagraph.hotkeys)
        start_uid = random.randint(0, max(0,n_uids-1))  # Pick random UID to start from
        all_uids = np.roll(np.arange(0, n_uids), start_uid).tolist()

        non_active_uids = set(all_uids) - active_uids - new_top_uids
        bt.logging.debug(f"Visiting UIDs for new models")
        n_new_models = self.visit_uids(
                new_metagraph,
                new_metadata,
                non_active_uids,
                prio=PRIO_NEW_MODEL,
        )

        # We need all models te be re-evaluated periodically, to make sure they
        # can win on changing competition parameters or advantage factors.
        # This is synchronized between validators by using UTC time, in order
        # to keep their weights in sync as much as possible. By testing only a
        # few extra models at a time, the weight setting interval is kept low
        # (previously there was a "test all models" event which took way longer
        # than a regular eval loop).
        interval = constants.GEN_MODEL_RETRY_INTERVAL
        force_eval_frac = float(int(now)%interval)/interval
        force_eval_until_uid = int(force_eval_frac*constants.SUBNET_N_UIDS)
        if self.force_eval_until_uid_last is None:
            # note regarding negative modulo in python: -1%10==9
            self.force_eval_until_uid_last = (force_eval_until_uid-constants.MODEL_RETRY_MAX_N_PER_LOOP)%constants.SUBNET_N_UIDS
        bt.logging.info(f"Forcing re-evaluation of models with {self.force_eval_until_uid_last} <= UID < {force_eval_until_uid}")

        if self.force_eval_until_uid_last < force_eval_until_uid:
            revisit_uids = np.arange(self.force_eval_until_uid_last, force_eval_until_uid).tolist()
        else:
            revisit_uids = ( np.arange(self.force_eval_until_uid_last, constants.SUBNET_N_UIDS).tolist() +
                            np.arange(0, force_eval_until_uid).tolist() )
        bt.logging.debug(f"Revisiting UIDs: {revisit_uids}")
        n_revisit_models = self.visit_uids(
                new_metagraph,
                new_metadata,
                revisit_uids,
                prio=PRIO_REVISIT,
                retry_interval=constants.GEN_MODEL_RETRY_INTERVAL//2,
        )
        self.force_eval_until_uid_last = force_eval_until_uid

    def update_dynamic_config(self):
        now = time.time()
        if self.last_cfg_fetch is not None and \
                (now - self.last_cfg_fetch) < constants.CFG_FETCH_INTERVAL:
            return
        self.last_cfg_fetch = now

        # Competition info
        url = constants.COMPETITIONS_URL
        comps = None
        if self.uid is not None:
            # Check if a dedicated competitions-{uid}.json is available.
            # This is used to allow targeted tweaks e.g. in case of issues with transformer overrides.
            vali_url = url.replace('.json',f'-{self.uid}.json')
            comps = competitions.load_competitions(vali_url,warn_failure=False)
        if comps is None:
            # Load regular competition info.
            comps = competitions.load_competitions(url)
        if comps is not None:
            with self.state_lock:
                self.competitions = comps
                self.model_updater.set_competitions(self.competitions)

        # Hall of fame
        try:
            req = requests.get(constants.HOF_URL)
            req.raise_for_status()
            with self.state_lock:
                self.hall_of_fame = req.json()
            bt.logging.info(f"Fetched hall of fame content, containing {len(self.hall_of_fame)} entries")
        except Exception as e:
            bt.logging.error(f"Failed to fetch hall of fame: {e}")

        self.save_state()

    def update_thread_func(self):
        """
        Updates all remote metadata: chain, competitions, hall of fame.
        Fetches all models needed for evaluation.
        Deletes models that are not needed, subject to diskspace requirements.
        """

        # Timestamp when we last retried models with incentive we've already dropped
        self.uid_last_retried_ts = dict()
        # Timestamp when we last fetched hall of fame
        self.last_cfg_fetch = None
        # Timestamp when we last updated chain
        self.last_chain_update = None

        while not self.stop_event.is_set():
            try:
                self.clean_models()
            except Exception as e:
                bt.logging.error(f"Error cleaning models: {e}, {traceback.format_exc()}")

            try:
                self.update_dynamic_config()
            except Exception as e:
                bt.logging.error(f"Failed to update dynamic config: {e} \n {traceback.format_exc()}")

            try:
                self.update_chain()
            except Exception as e:
                bt.logging.error(f"Failed to update chain data: {e} \n {traceback.format_exc()}")

            # Regular sleep, we are running in a separate thread
            self.update_thread_ts = time.time()
            time.sleep(60)

        bt.logging.info("Exiting update models loop.")

    def clean_models(self):
        """Cleans up models that are no longer referenced."""

        base_dir = pathlib.Path(self.local_store.base_dir)
        if not base_dir.exists():
            # Nothing to clean
            return

        state = disk_utils.storage_state(base_dir=base_dir,config=self.config)
        if state['gb_to_delete'] == 0:
            bt.logging.info(f"Skipping cleanup of stale models; {state['usage_str']}")
            return

        bt.logging.info(f"Starting model cleanup, deleting at least {state['gb_to_delete']} GB; {state['usage_str']}")

        # Get a mapping of all hotkeys to model ids.
        hotkey_to_model_id = {
            hotkey: metadata.id
            for hotkey, metadata in self.hk_metadata.items() if metadata is not None
        }

        # Find all hotkeys that are currently being evaluated or pending eval.
        uids_to_keep = self.get_all_active_uids()

        with self.metagraph_lock:
            # Injected models might not be in metagraph
            hotkeys_to_keep = set([
                self.metagraph.hotkeys[uid]
                    for uid in uids_to_keep if uid < len(self.metagraph.hotkeys)
            ])

        # Only keep those hotkeys.
        evaluated_hotkeys_to_model_id = {
            hotkey: model_id
            for hotkey, model_id in hotkey_to_model_id.items()
            if hotkey in hotkeys_to_keep
        }

        self.local_store.delete_unreferenced_models(
            valid_models_by_hotkey=evaluated_hotkeys_to_model_id,
            grace_period_seconds=300,
            gb_to_delete=state['gb_to_delete'],
        )

    async def try_set_weights(self, ttl: int):
        """Sets the weights on the chain with ttl, without raising exceptions if it times out."""
        if self.last_weights_set is not None and time.time() - self.last_weights_set < constants.WEIGHT_SET_MIN_INTERVAL:
            bt.logging.debug(f'Not setting weights; {time.time()} - {self.last_weights_set} < {constants.WEIGHT_SET_MIN_INTERVAL}')
            return

        self.weights.nan_to_num(0.0)
        weight_sum = self.weights.sum().item()
        if weight_sum < 1e-5:
            bt.logging.warning(f'Weights all zero, not setting')
            return

        ws, ui = self.weights.topk(len(self.weights))
        table = Table(title="All non-zero weights")
        table.add_column("uid", justify="right", style="cyan", no_wrap=True)
        table.add_column("weight", style="magenta")
        for index, weight in list(zip(ui.tolist(), ws.tolist())):
            if weight == 0:
                continue
            table.add_row(str(index), str(round(weight, 4)))
        console = Console()
        console.print(table)

        # always get a new subtensor instance for weight setting
        st = btlite.get_subtensor()
        if st is None:
            bt.logging.error(f'Could not create subtensor, cannot set weights')
            return

        try:
            bt.logging.info(f"Setting weights.")
            call = btlite.set_weights_retry(
                subtensor=st,
                hotkey=self.wallet.hotkey,
                uids=self.metagraph.uids,
                netuid=constants.SUBNET_UID,
                weights=self.weights[:len(self.metagraph.uids)],
                wait_for_inclusion=True,
                version_key=constants.weights_version_key,
            )
            ret,msg = await asyncio.wait_for(call, ttl)
            if ret:
                self.last_weights_set = time.time()
                bt.logging.info(f"Finished setting weights at {self.last_weights_set}.")
            else:
                bt.logging.warning(f"Failed to set weights: {msg}")
        except asyncio.TimeoutError:
            bt.logging.error(f"Failed to set weights after {ttl} seconds")
        except Exception as e:
            bt.logging.warning(f"Failed to set weights: {e}, {traceback.format_exc()}")


    async def try_run_step(self, ttl: int):
        """Runs a step with ttl, without raising exceptions if it times out."""
        try:
            bt.logging.trace(f"Running step with ttl {ttl}.")
            t0 = time.time()
            await asyncio.wait_for(self.run_step(), ttl)
            bt.logging.trace(f"Finished running step in {time.time()-t0:.1f}s.")
        except asyncio.TimeoutError:
            bt.logging.error(f"Failed to run step after {ttl} seconds")

    def get_reward_weights(self):
        reward_weights = torch.zeros_like(self.weights)
        with self.metagraph_lock:
            metagraph = copy.deepcopy(self.metagraph)
        current_block = metagraph.block.item()

        for uid, hotkey in enumerate(metagraph.hotkeys):
            if hotkey not in self.hall_of_fame:
                continue
            try:
                for entry in self.hall_of_fame[hotkey]:
                    delta_blocks = current_block - entry['block']
                    delta_epochs = delta_blocks / constants.blocks_per_epoch
                    iv = entry['reward'] / constants.REWARDS_IV_FACTOR
                    reward_weights[uid] += iv * constants.REWARDS_DECAY_FACTOR**delta_epochs
                    bt.logging.info(f"Rewarding UID {uid} / {hotkey} with weight {reward_weights[uid]:.04f} for '{entry.get('desc','')}'")
            except Exception as e:
                bt.logging.warning(f"Reward computation for UID {uid} / {hotkey} failed ({e})")

        # Make sure total reward weight does not exceed BOUNTIES_MAX_FRACTION
        total_weight = reward_weights.sum().item()
        if total_weight > constants.REWARDS_MAX_FRACTION:
            reward_weights *= constants.REWARDS_MAX_FRACTION / total_weight
            total_weight = constants.REWARDS_MAX_FRACTION

        return reward_weights, total_weight

    def update_weights(self):
        '''
        Update self.weights, based on internal cstate
        Up to constants.REWARDS_MAX_FRACTION part of the total weight will be based
        on the hall of fame rewards settings.
        '''
        new_weights = torch.zeros_like(self.weights)
        reward_weights, reward_sum = self.get_reward_weights()

        # Scale model weights down by (1 - reward_sum)
        for cname, cparams in self.competitions.items():
            if cname not in self.cstate:
                bt.logging.warning(f"No evaluations in competition {cname}")
                continue

            for uid, weight in self.cstate[cname]['uids_weight'].items():
                new_weights[uid] = (1 - reward_sum) * cparams['reward'] * weight

        # Add bounties
        new_weights += reward_weights
        # Normalize total, which should in principle already be the case
        new_weights /= new_weights.sum()

        # First time around, use weights without EMA
        if self.weights.count_nonzero().item() == 0:
            self.weights = new_weights
        else:
            self.weights = constants.weight_alpha * self.weights + (1 - constants.weight_alpha) * new_weights
        self.weights = self.weights.nan_to_num(0.0)

    def load_benchmark_config(self):
        if not os.path.exists(self.BENCHMARK_FILENAME):
            return {}
        try:
            with open(self.BENCHMARK_FILENAME, 'rb') as f:
                d = {int(k): v for k, v in orjson.loads(f.read()).items()}
            bt.logging.info(f"Loaded benchmark config with {len(d)} items")
            return d
        except Exception as e:
            bt.logging.warning(f"Failed to load benchmark config: {e}")
        return {}

    async def run_step(self):
        bt.logging.debug(f'run_step() @ current block {self.current_block}')
        self.inject_models()
        t0 = time.time()

        # Collect uid evaluation data here
        self.step_uid_log = dict()

        # Currently, all competitions use the same dataset.
        # Fetch samples that are shared between all competitions
        try:
            dataloader = dataset.SubsetFineWebEdu2Loader(
                batch_size=1,
                num_pages=0,
                tokenizer=None,
                pack=False
            )
        except requests.exceptions.RequestException as e:
            bt.logging.warning(f"Exception instantiating dataloader: {e}. Waiting one minute before retrying.")
            await asyncio.sleep(60)
            return

        samples = dataloader.fetch_data_to_rows(constants.n_eval_pages)
        if len(samples) == 0:
            bt.logging.warning(f"No samples to eval. Waiting one minute before retrying.")
            await asyncio.sleep(60)
            return

        n_models_evaluated = 0
        t_per_competition = constants.TTL_RUN_STEP/len(self.competitions)
        for cname in self.competitions:
            ts_expire = time.time() + t_per_competition
            n_models_evaluated += self.run_step_for_competition(cname, dataloader, ts_expire=ts_expire)

        if n_models_evaluated == 0:
            bt.logging.debug("No uids to eval. Waiting 2 minutes to download some models.")
            await asyncio.sleep(120)
            return

        self.update_weights()

        self.save_state()

        # Log to screen and wandb.
        pages_desc = [f'{cfg_name}_{num_rows}_{split}' for cfg_name, num_rows, split in dataloader.pages]
        step_time = time.time() - t0
        self.log_step(pages=pages_desc, step_time=step_time)

        # Increment the number of completed run steps by 1
        self.run_step_count += 1

    def run_step_for_competition(self, cname, dataloader, ts_expire=None):
        """
        Run step for one of the competitions
        Return number of uids evaluated
        """

        cinfo = self.competitions[cname]
        pool_size = cinfo.get('pool_size', constants.DEFAULT_POOL_SIZE)
        pend_size = cinfo.get('pend_size', constants.DEFAULT_PEND_SIZE)

        with self.state_lock:
            # Initialize competition state
            if cname not in self.cstate:
                self.cstate[cname] = dict(
                    uids_pool=[],
                    uids_pending={},
                )
            cstate = self.cstate[cname]

            # Select at most pool_size + pend_size uids, sorted on priority number
            pend = cstate['uids_pending']
            cur_pool = cstate['uids_pool']
            bt.logging.debug(f"Competition {cname} pool: {cur_pool}, pending: {pend}")
            pend.update({uid: PRIO_POOL for uid in cur_pool})
            pend = [(prio, uid) for uid, prio in pend.items()]
            pend.sort(reverse=True)
            uids_pool = [uid for (prio, uid) in pend[:pool_size+pend_size]]
            picked = set(uids_pool) - set(cur_pool)
            if len(picked):
                bt.logging.debug(f"Picked: {picked}")

            # Update pool/pending state
            cstate['uids_pending'] = {}
            cstate['uids_pool'] = uids_pool

        if len(uids_pool) == 0:
            with self.state_lock:
                cstate['uids_weight'] = {}
            bt.logging.debug(f"No miners participating in competition {cname}")
            return 0

        bt.logging.debug(f"Evaluating competition {cname} with uids {uids_pool} on {len(dataloader.buffer)} samples")


        # Competition-wide tokenizer
        n_batches = len(dataloader.buffer)
        batches = None
        batches_max_token_id = None
        if 'tokenizer' in cinfo:
            # Competition-wide (forced or default) tokenizer --> fixed sequence length
            batches = dataloader.tokenize(
                    cinfo['tokenizer'],
                    max_len=cinfo.get('max_sequence_len', constants.MAX_SEQUENCE_LEN),
                    max_invalid=cinfo.get('max_tokenize_fails', constants.MAX_TOKENIZE_FAILS)
            )
            batches_max_token_id = max(
                [max(b[0]) for b in batches if b is not None]
            )


        # Compute model losses on batches.
        losses_per_uid = {uid: None for uid in uids_pool}
        losses_pt_per_uid = {uid: None for uid in uids_pool}
        avg_sample_len_per_uid = {uid: None for uid in uids_pool}
        model_geometry_per_uid = {uid: {} for uid in uids_pool}
        uid_to_label = {uid: '' for uid in uids_pool}
        uid_to_block = {uid: 1<<31 for uid in uids_pool}
        n_evaluated = 0
        for uid in uids_pool:
            if ts_expire is not None and time.time() > ts_expire:
                bt.logging.warning("Model eval loop taking too long, stopping loop")
                break

            bt.logging.trace(f"Computing model losses for uid {uid}.")
            metadata = self.get_uid_metadata(uid)

            losses_per_uid[uid] = [math.inf]*n_batches
            losses_pt_per_uid[uid] = losses_per_uid[uid].copy()
            if metadata is None:
                bt.logging.debug(f"Unable to load metadata for {uid}. Setting loss to infinity.")
                continue
            try:
                uid_to_block[uid] = metadata.block if metadata.block is not None else 1<<31
                uid_to_label[uid] = metadata.id.format_label()
                vminfo = psutil.virtual_memory()
                cpu_mem_gb_total = vminfo.total>>30
                cpu_mem_gb_free = vminfo.available>>30
                bt.logging.debug(f"Evaluating uid {uid} ({uid_to_label[uid]}) from block {uid_to_block[uid]}, {cpu_mem_gb_free}/{cpu_mem_gb_total} Gb free")
                # Get model tokenizer if no competition-wide tokenizer is set
                mdl_batches = batches
                max_token_id = batches_max_token_id
                model_path = disk_utils.get_local_model_snapshot_dir(
                        self.local_store.base_dir,
                        metadata.hotkey,
                        metadata.id) if metadata.path is None else metadata.path
                tokenizer_json = os.path.join(model_path,'tokenizer.json')
                if not os.path.exists(tokenizer_json):
                    # Assume tokenizer.json indicates an embedded tokenizer is available.
                    if mdl_batches is None:
                        raise ModelIssue(f'No default tokenizer and no model tokenizer available')
                elif mdl_batches is None or cinfo.get('free_tokenizer',False):
                    max_len = cinfo.get('max_sequence_len', None)
                    if max_len is None:
                        raise ModelIssue(f"Unable to determine max sequence length")
                    try:
                        new_mdl_batches = None
                        new_mdl_batches = dataloader.tokenize(
                                model_path,
                                max_len=max_len,
                                max_invalid=cinfo.get('max_tokenize_fails', constants.MAX_TOKENIZE_FAILS)
                        )
                        mdl_batches = new_mdl_batches
                        max_token_id = max(
                            [max(b[0]) for b in mdl_batches if b is not None]
                        )
                        bt.logging.info("Using model-supplied tokenizer")
                    except Exception as e:
                        if new_mdl_batches is None:
                            if mdl_batches is None:
                                raise ModelIssue(f'No default tokenizer and no model tokenizer available: {e}')
                            bt.logging.info(f"Using default tokenizer {cinfo.get('tokenizer', 'unknown')}, because {e}")

                if mdl_batches is None:
                    # We should never arrive here
                    raise Exception("No tokenizer available (no default and not supplied in model)")


                eval_results = utils.run_in_subprocess(
                    functools.partial(
                        check_and_compute_losses,
                        local_store=self.local_store,
                        metadata=metadata,
                        competition_info=cinfo,
                        batches=mdl_batches,
                        max_token_id=max_token_id,
                        device=self.config.device,
                    ),
                    ttl=constants.TTL_MODEL_EVAL,
                    mode="spawn",
                    expected_errors={"ModelIssue"},
                )

                losses = eval_results['losses']
                losses_pt = eval_results['losses_pt']
                n_evaluated += 1

                losses_per_uid[uid] = losses
                losses_pt_per_uid[uid] = losses_pt
                avg_sample_len_per_uid[uid] = eval_results['avg_sample_length']
                model_geometry_per_uid[uid] = eval_results['model_geometry']
                bt.logging.debug(f"Losses for uid:{uid}, per token: {naninf_mean(losses_pt):.03f} +- {naninf_std(losses_pt):.03f}, sum {naninf_mean(losses):.01f} +- {naninf_std(losses):.01f}, avg sample len: {eval_results['avg_sample_length']:.01f}")

            except ModelIssue as e:
                bt.logging.info(
                    f'Model issue for uid {uid}, disqualifying: {e}'
                )
            except Exception as e:
                bt.logging.error(
                    f"Error in eval loop: {e}. Setting losses for uid: {uid} to infinity.\n{traceback.format_exc()}"
                )
                if transformers.__version__ != TRANSFORMERS_VERSION_OPTIMAL:
                    bt.logging.error(f'Please run with transformers version {TRANSFORMERS_VERSION_OPTIMAL} (currently running {transformers.__version__}) before reporting issues.')

        group_samples = cinfo.get('group_samples', 1)
        if group_samples > 1:
            losses_per_uid = validation.group_samples(losses_per_uid, group_samples)

        win_info = validation.compute_wins(
                losses_per_uid,
                uid_to_block,
                self.current_block,
                cinfo.get('advantage_initial', constants.advantage_initial),
                cinfo.get('advantage_decay', constants.advantage_decay_per_epoch),
        )
        if 'win_rate' not in win_info:
            bt.logging.warning("compute_wins() returned no result")
            return 0

        # Skew weight distribution towards models with high win_rate
        win_rate = np.array([
            win_info['win_rate'][uid] if uid in win_info['win_rate'] else 0
                for uid in uids_pool
        ])
        skew_factor = cinfo.get('weight_skew_factor',constants.WEIGHT_SKEW_FACTOR)
        model_weights = win_rate**skew_factor
        weight_sum = np.sum(model_weights)
        if weight_sum:
            model_weights /= weight_sum
        model_weights = {uid: weight for uid, weight in zip(uids_pool, model_weights)}

        # Sort models by weight / win_rate, keep pool_size entries
        win_rate_indices = win_rate.argsort()
        sorted_uids = [uids_pool[i] for i in win_rate_indices]
        new_uids_pool = sorted_uids[-pool_size:]
        bt.logging.info(f'selected {pool_size} winning models: {new_uids_pool}')

        dropped_uids = sorted_uids[:-pool_size]
        if len(dropped_uids):
            bt.logging.info(f'Dropping uids {dropped_uids} from pool')
        for uid in dropped_uids:
            if uid in self.benchmark_cfg:
                continue
            hk = self.get_uid_hotkey(uid)
            metadata = self.get_uid_metadata(uid)
            if hk is None or metadata is None:
                continue
            if metadata.id is None:
                continue
            state = disk_utils.storage_state(
                base_dir=self.local_store.base_dir,
                config=self.config
            )
            if state['gb_space_left'] == 0:
                bt.logging.debug(f"deleting model for UID {uid}; {state['usage_str']}")
                self.local_store.delete_model(hk, metadata.id)

        # Update state: weights and which uids to keep for next run
        with self.state_lock:
            self.cstate[cname]['uids_weight'] = model_weights
            self.cstate[cname]['uids_pool'] = new_uids_pool

        win_matrix = win_info.get('matrix', None)
        if win_matrix is not None:
            self.print_win_matrix(win_info['matrix'], uid_to_label, competition=cname)

        # Update step log
        wins = win_info.get('wins', {})
        win_rate = win_info.get('win_rate', {})
        win_abs_rate = win_info.get('win_abs_rate', {})
        advantage_factors = win_info.get('advantage_factors', {})
        for uid in uids_pool:
            self.step_uid_log[uid] = {
                "uid": uid,
                "geometry": model_geometry_per_uid.get(uid,{}),
                "competition": cname,
                "label": uid_to_label.get(uid, ''),
                "block": uid_to_block.get(uid, 1<<31),
                "losses": losses_per_uid[uid],
                "n_samples": naninf_count(losses_per_uid[uid]),
                "n_inf": np.sum(np.isinf(losses_per_uid[uid])),
                "avg_sample_len": avg_sample_len_per_uid[uid],
                "loss_pt_avg": naninf_mean(losses_pt_per_uid[uid]),
                "loss_pt_std": naninf_std(losses_pt_per_uid[uid]),
                "loss_sum_avg": naninf_mean(losses_per_uid[uid]),
                "loss_sum_std": naninf_std(losses_per_uid[uid]),
                "adv_factor": 100*(1-advantage_factors.get(uid,1)),
                "win_rate": win_rate.get(uid, 0),
                "win_abs_rate": win_abs_rate.get(uid, 0),
                "win_total": wins.get(uid, 0),
                "win_matrix_row": win_matrix.get(uid, None) if win_matrix else None
            }

        return n_evaluated

    def get_uid_hotkey(self, uid):
        with self.metagraph_lock:
            if uid < len(self.metagraph.hotkeys):
                return self.metagraph.hotkeys[uid]
            else:
                return None

    def get_uid_metadata(self, uid):
        metadata = None
        if uid in self.benchmark_cfg:
            # Model data from dynamic config
            metadata = Container()
            bcfg = self.benchmark_cfg[uid]
            metadata.hotkey = bcfg.get("hotkey", "xxx")
            metadata.block = bcfg.get("block", 1<<31)
            metadata.path = bcfg.get('path', 'please specify path')
            metadata.id = ModelId.dummy(bcfg.get('label', os.path.split(metadata.path)[-1]))
        elif uid < len(self.metagraph.hotkeys):
            # Model from chain
            hotkey = self.get_uid_hotkey(uid)
            chain_data = self.hk_metadata.get(hotkey, None)
            if chain_data is not None:
                metadata = Container()
                metadata.hotkey = hotkey
                metadata.id = chain_data.id
                metadata.block = chain_data.block
                metadata.path = None
        return metadata

    def inject_models(self):
        self.benchmark_cfg = self.load_benchmark_config()
        with self.state_lock:
            for uid, binfo in self.benchmark_cfg.items():
                competition = binfo.get('competition', '')
                if competition not in self.cstate:
                    bt.logging.info(f"Injected model {uid} competition '{competition}' unknown")
                    continue
                ci = self.cstate[competition]
                if uid not in ci['uids_pool'] and uid not in ci['uids_pending']:
                    bt.logging.info(f"Injecting model {uid} into competition '{competition}'")
                    ci['uids_pending'][uid] = PRIO_INJECT

    def print_win_matrix(self, matrix, uid_to_label={}, show_delta_loss=False, competition='?'):
        if show_delta_loss:
            title = f"Win matrix compt {competition}, true wins/adv wins/avg delta loss"
        else:
            title = f"Win matrix compt {competition}, true wins/adv wins"
        table = Table(title=title)
        table.add_column("win \ lose", justify="right", style="cyan", no_wrap=True)
        for uid in matrix:
            table.add_column(f'UID {uid}')
        for uid_a,row in matrix.items():
            label = uid_to_label.get(uid_a, '')
            vals = [f'{label} UID {uid_a}']
            for uid_b,wins in row.items():
                val = '?'
                if uid_a==uid_b:
                    val = '...'
                elif wins is not None:
                    if show_delta_loss:
                        val = f'{wins["wins"]}/{wins["wins_adv"]}/{wins["loss"]:.01f}'
                    else:
                        val = f'{wins["wins"]}/{wins["wins_adv"]}'
                vals.append(val)
            table.add_row(*vals)
        console = Console()
        console.print(table)

    def log_step(self, **kwargs):
        """Logs the results of the step to the console and wandb (if enabled)."""
        # Build step log
        step_log = {
            "timestamp": time.time(),
            'block': self.current_block,
            "uids": [int(uid) for uid in self.step_uid_log.keys()],
            "competitions": self.competitions,
            "uid_data": {},
        }
        step_log.update(kwargs)
        for uid, info in self.step_uid_log.items():
            info['weight'] = self.weights[uid].item()
            step_log['uid_data'][str(uid)] = info

        if self.config.save_step_json:
            try:
                with open(self.config.save_step_json, 'wb') as f:
                    f.write(orjson.dumps(step_log, option=orjson.OPT_NON_STR_KEYS|orjson.OPT_SERIALIZE_NUMPY))
            except Exception as e:
                bt.logging.warning(f"Failed to write step json to {self.config.save_step_json}: {e}")

        table = Table(title="Step")
        table.add_column("uid", justify="right", style="cyan", no_wrap=True)
        table.add_column("compt", style="magenta")
        table.add_column("avg_pt_loss", style="magenta")
        table.add_column("avg_sum_loss", style="magenta")
        table.add_column("avg_slen", style="magenta")
        table.add_column("adv_factor(%)", style="magenta")
        table.add_column("win_rate", style="magenta")
        table.add_column("win_abs_rate", style="magenta")
        table.add_column("win_total", style="magenta")
        table.add_column("weights", style="magenta")
        table.add_column("block", style="magenta")
        for uid, d in step_log['uid_data'].items():
            try:
                table.add_row(
                    f"{uid} {d.get('label','')}",
                    str(d['competition']),
                    f"{d['loss_pt_avg']:.04f}+-{d['loss_pt_std']:.04f}",
                    f"{d['loss_sum_avg']:.01f}+-{d['loss_sum_std']:.01f}",
                    f"{d['avg_sample_len']:.01f}",
                    f"{d['adv_factor']:.03f} of {100*constants.advantage_initial:.03f}",
                    str(round(d["win_rate"], 4)),
                    str(round(d["win_abs_rate"], 4)),
                    str(d["win_total"]),
                    str(round(d['weight'], 4)),
                    str(d["block"]),
                )
            except:
                pass
        console = Console()
        console.print(table)

        ws, ui = self.weights.topk(len(self.weights))
        table = Table(title="Weights > 0.001")
        table.add_column("uid", justify="right", style="cyan", no_wrap=True)
        table.add_column("weight", style="magenta")
        for index, weight in list(zip(ui.tolist(), ws.tolist())):
            if weight > 0.001:
                table.add_row(str(index), str(round(weight, 4)))
        console = Console()
        console.print(table)

        if self.config.wandb.on and not self.config.offline:
            # If we have already completed X steps then we will complete the current wandb run and make a new one.
            if (
                self.run_step_count
                and self.run_step_count % constants.MAX_RUN_STEPS_PER_WANDB_RUN == 0
            ):
                bt.logging.trace(
                    f"Validator has completed {self.run_step_count} run steps. Creating a new wandb run."
                )
                self.wandb_run.finish()
                self.new_wandb_run()

            original_format_json = orjson.dumps(step_log, option=orjson.OPT_NON_STR_KEYS|orjson.OPT_SERIALIZE_NUMPY).decode()
            uids = step_log["uids"]
            uid_data = step_log["uid_data"]

            # Create a new dictionary with the required format
            graphed_data = {
                "time": time.time(),
                "block": self.metagraph.block.item(),
                "uid_data": {
                    str(uid): uid_data[str(uid)]["loss_sum_avg"] for uid in uids
                },
                "weight_data": {str(uid): self.weights[uid].item() for uid in uids},
            }
            bt.logging.trace("Logging to Wandb")
            self.wandb_run.log(
                {**graphed_data, "original_format_json": original_format_json},
                step=self.global_step,
            )

    def shutdown(self,signum,stackframe):
        print('SIGINT caught, exiting',file=sys.stderr)
        if self.wandb_run:
            bt.logging.info("gracefully closing the wandb run...")
            self.wandb_run.finish()
        try:
            if self.update_thread.is_alive():
                pid = self.update_thread.native_id
                print(f'killing update thread, PID={pid}',file=sys.stderr)
                os.kill(pid,signal.SIGTERM)
            print(f'joining update thread',file=sys.stderr)
            self.update_thread.join()
        except Exception as e:
            print(f'exception trying to stop update_thread: {e}')
        sys.exit(-1)

    async def run(self):
        """Runs the validator loop, which continuously evaluates models and sets weights."""
        # Install signal handler for clean shutdown.
        signal.signal(signal.SIGINT, self.shutdown)
        # Give update thread some time to fetch initial state from chain.
        await asyncio.sleep(60)
        while True:
            try:
                self.current_block = self.metagraph.block.item()
                await self.try_run_step(ttl=2*constants.TTL_RUN_STEP)
                self.save_state()
                self.global_step += 1

                if not self.config.dont_set_weights and not self.config.offline:
                    await self.try_set_weights(ttl=60)
                else:
                    bt.logging.warning(f'Not setting weights due to config: {self.config.dont_set_weights} or {self.config.offline}')
                self.last_epoch = self.metagraph.block.item()
                self.epoch_step += 1

            except Exception as e:
                bt.logging.error(
                    f"Error in validator loop \n {e} \n {traceback.format_exc()}"
                )

def check_and_compute_losses(
        local_store=None,
        metadata=None,
        competition_info=None,
        batches=None,
        max_token_id=None,
        device=None,
    ):
    cinfo = competition_info
    model_i = local_store.retrieve_model(metadata.hotkey, metadata.id, path=metadata.path)
    mdl_allowed, reason = competitions.validate_model_constraints(model_i.pt_model, cinfo)
    if not mdl_allowed:
        raise ModelIssue(f"Model violates competition {cname} constraints: {reason}")
    allow_sliced = False
    model_type = type(model_i.pt_model).__name__
    if 'Sliced' in model_type:
        # Test the exact model type name to check whether slicing is allowed by config:
        allow_sliced = model_type in cinfo['model_types']

    embed_size = None
    try:
        embed_size = model_i.pt_model.model.embed_tokens.weight.shape[0]
    except Exception as e:
        # Currently supported models should have the queried parameter, but in case they don't, just skip this check.
        bt.logging.warning(f'could not find embed size, skipping check: {e}')

    if embed_size:
        if max_token_id>=embed_size:
            raise ModelIssue(f"Vocabulary size mismatch between tokenizer and model: {max_token_id} >= {embed_size}")
    else:
        embed_size = max_token_id

    model_geometry = {
        'n_parameters':model_i.pt_model.num_parameters(),
        'n_layers':getattr(model_i.pt_model.config,'num_hidden_layers',0),
        'embed_size':embed_size,
        'intermediate_size':getattr(model_i.pt_model.config,'intermediate_size',0),
        'hidden_size':getattr(model_i.pt_model.config,'hidden_size',0),
    }

    losses = validation.compute_losses(model_i.pt_model,allow_sliced,batches,device)
    losses_pt = [loss_sum / len(batch[0]) if batch is not None else math.inf for loss_sum, batch in zip(losses, batches)]
    sample_lengths = [len(batch[0]) for batch in batches if batch is not None]
    avg_sample_length = 0 if len(sample_lengths) == 0 else np.mean(sample_lengths)

    return {
        'losses':losses,
        'losses_pt':losses_pt,
        'avg_sample_length':avg_sample_length,
        'model_geometry':model_geometry,
    }

def assert_cuda():
    if transformers.utils.is_flash_attn_2_available():
        bt.logging.warning('Flash Attention 2 is available, according to transformers.')
        return
    import importlib
    bt.logging.error('Flash Attention 2 is not available, according to transformers. Possible issues:')
    if not transformers.utils.is_torch_available():
        bt.logging.error('torch is not available, according to transformers')
    if not transformers.utils.is_torch_mlu_available():
        bt.logging.error('torch_mlu is not available, according to transformers')
    if not torch.cuda.is_available():
        bt.logging.error('cuda is not available, according to torch')
    bt.logging.error(f'torch.version.cuda = {torch.version.cuda}')
    flash_attn_version = importlib.metadata.version("flash_attn")
    bt.logging.error(f'flash_attn version = {flash_attn_version}')
    sys.exit(-1)

if __name__ == "__main__":
    if Version(transformers.__version__) < Version(TRANSFORMERS_VERSION_MIN):
        bt.logging.error(f"Transformers version >= {TRANSFORMERS_VERSION_MIN} required")
        sys.exit()

    assert_cuda()

    # Set an output width explicitly for rich table output; on pm2 tables get
    # squished if we don't.
    try:
        width = os.get_terminal_size().columns
    except:
        width = 0
    os.environ['COLUMNS'] = str(max(200,width))

    asyncio.run(Validator().run())
