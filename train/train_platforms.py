import os
import glob

class TrainPlatform:
    def __init__(self, save_dir, *args, **kwargs):
        self.path, file = os.path.split(save_dir)
        self.name = kwargs.get('name', file)

    def report_scalar(self, name, value, iteration, group_name=None):
        pass

    def report_media(self, title, series, iteration, local_path):
        pass

    def report_args(self, args, name):
        pass

    def close(self):
        pass


class ClearmlPlatform(TrainPlatform):
    def __init__(self, save_dir):
        from clearml import Task
        path, name = os.path.split(save_dir)
        self.task = Task.init(project_name='my_project',
                              task_name=name)
        self.logger = self.task.get_logger()

    def report_scalar(self, name, value, iteration, group_name):
        self.logger.report_scalar(title=group_name, series=name, iteration=iteration, value=value)

    def report_media(self, title, series, iteration, local_path):
        self.logger.report_media(title=title, series=series, iteration=iteration, local_path=local_path)

    def report_args(self, args, name):
        self.task.connect(args, name=name)

    def close(self):
        self.task.close()


class TensorboardPlatform(TrainPlatform):
    def __init__(self, save_dir):
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(log_dir=save_dir)

    def report_scalar(self, name, value, iteration, group_name=None):
        self.writer.add_scalar(f'{group_name}/{name}', value, iteration)

    def close(self):
        self.writer.close()


class NoPlatform(TrainPlatform):
    def __init__(self, save_dir):
        pass


class WandBPlatform(TrainPlatform):
    import wandb
    def __init__(self, save_dir, resume='allow', config=None, debug=False, *args, **kwargs):
        super().__init__(save_dir, args, kwargs)
        if debug:
            self.debug = True
        else:
            self.debug = False
            self.wandb.login(host=os.getenv("WANDB_BASE_URL"), key=os.getenv("WANDB_API_KEY"))
            # check if an experiment with the same id is already running
            api = self.wandb.Api()
            runs = api.runs(path=f'inbars_projects/GMDM')
            for run in runs:
                print(run.name, run.state)
                if run.name == self.name and run.state == 'running':
                    raise Exception(f'Experiment with name {self.name} is already running')
            self.wandb.init(
                project='GMDM',
                name=self.name,
                id=self.name,  
                resume=resume, # in order to send continued runs to the same record
                entity='inbars_projects',
                save_code=True,
                config=config)  # config can also be sent via report_args()
            self.wandb.run.log_code(".")

    def report_scalar(self, name, value, iteration, group_name=None):
        if self.debug:
            return
        self.wandb.log({name: value}, step=iteration)

    def report_media(self, title, series, iteration, local_path):
        if self.debug:
            return
        files = glob.glob(f'{local_path}/*.mp4')
        self.wandb.log({series: [self.wandb.Video(file, format='mp4', fps=20) for file in files]}, step=iteration)

    def report_args(self, args, name):
        if self.debug:
            return
        self.wandb.config.update(args)  # , allow_val_change=True) # use allow_val_change ONLY if you want to change existing args (e.g., overwrite)

    def watch_model(self, *args, **kwargs):
        if self.debug:
            return
        self.wandb.watch(args, kwargs)

    def close(self):
        if self.debug:
            return
        self.wandb.finish()