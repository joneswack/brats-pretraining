import logging
import time
import os

import pandas as pd

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

class LogReader():
    def __init__(self, path):
        super(LogReader, self).__init__()
        
        self.path = path

    def load_run(self, run_index=0, test=False) -> EventAccumulator:
        runs = sorted(os.listdir(self.path), reverse=True)
        suffix = 'test' if test else 'train'
        
        selected_path = os.path.join(self.path, runs[run_index], suffix)

        event_acc = EventAccumulator(selected_path)
        logging.warning('Starting to load the event file in {}. May take a while.'.format(selected_path))
        t0 = time.time()
        event_acc.Reload()
        t_diff = time.time() - t0
        logging.info('Loading from {} completed in {} seconds'.format(selected_path, t_diff))
        
        return event_acc
        
    def get_available_scalars(self, events):
        return events.scalars.Keys()
    
    def get_scalar_times(self, events, scalar):
        scalar_summary = events.Scalars(scalar)
        return [step.wall_time for step in scalar_summary]

    def get_scalar_steps(self, events, scalar):
        scalar_summary = events.Scalars(scalar)
        return [step.step for step in scalar_summary]
    
    def get_scalar_values(self, events, scalar):
        scalar_summary = events.Scalars(scalar)
        return [step.value for step in scalar_summary]
    
    def get_df_from_scalar(self, events, scalar):
        scalar_dict = {
            'times': self.get_scalar_times(events, scalar),
            'steps': self.get_scalar_steps(events, scalar),
            'values': self.get_scalar_values(events, scalar)
        }
        
        return pd.DataFrame.from_dict(scalar_dict)