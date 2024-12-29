import os
import numpy as np
import matplotlib.pyplot as plt
from evo.core import metrics
from evo.tools import plot
from evo.core.trajectory import PoseTrajectory3D
from evo.core import sync
from evo.core import lie_algebra as lie

from src.utils.pose_trajectory import OutputTrajectory
from src.utils.Visualizer import CameraPoseVisualizer


class PoseErrorGlobalEvaler:
    """ Compute APE and RPE for full trajectory """

    def __init__(self):

        # define metrics
        self.ape_full_metric = metrics.APE(metrics.PoseRelation.full_transformation)
        self.ape_pos_metric = metrics.APE(metrics.PoseRelation.translation_part)
        self.ape_rot_metric = metrics.APE(metrics.PoseRelation.rotation_part)
        self.ape_deg_metric = metrics.APE(metrics.PoseRelation.rotation_angle_deg)
        self.rpe_full_metric = metrics.RPE(metrics.PoseRelation.full_transformation)
        self.rpe_pos_metric = metrics.RPE(metrics.PoseRelation.translation_part)
        self.rpe_rot_metric = metrics.RPE(metrics.PoseRelation.rotation_part)
        self.rpe_deg_metric = metrics.RPE(metrics.PoseRelation.rotation_angle_deg)

        self.ape_full_stats = None
        self.ape_pos_stats = None
        self.ape_rot_stats = None
        self.ape_deg_stats = None
        self.rpe_full_stats = None
        self.rpe_pos_stats = None
        self.rpe_rot_stats = None
        self.rpe_deg_stats = None

    def process_data(self, traj_ref, traj_est):
        """ process a pair of trajectories 
        
        Args:
            traj_ref: evo.core.trajectory.PoseTrajectory3D instance, reference trajectory
            traj_est: evo.core.trajectory.PoseTrajectory3D instance, estimated trajectory
        """
        data = (traj_ref, traj_est)

        # process data
        self.ape_full_metric.process_data(data)
        self.ape_pos_metric.process_data(data)
        self.ape_rot_metric.process_data(data)
        self.ape_deg_metric.process_data(data)
        self.rpe_full_metric.process_data(data)
        self.rpe_pos_metric.process_data(data)
        self.rpe_rot_metric.process_data(data)
        self.rpe_deg_metric.process_data(data)

        # get statistics
        # contains: rmse, mean, std, min, max, median, sse
        self.ape_full_stats = self.ape_full_metric.get_all_statistics()
        self.ape_pos_stats = self.ape_pos_metric.get_all_statistics()
        self.ape_rot_stats = self.ape_rot_metric.get_all_statistics()
        self.ape_deg_stats = self.ape_deg_metric.get_all_statistics()
        self.rpe_full_stats = self.rpe_full_metric.get_all_statistics()
        self.rpe_pos_stats = self.rpe_pos_metric.get_all_statistics()
        self.rpe_rot_stats = self.rpe_rot_metric.get_all_statistics()
        self.rpe_deg_stats = self.rpe_deg_metric.get_all_statistics()

    def get_statistics(self):
        return {
            f"global_APE_rmse": self.ape_full_stats["rmse"],
            f"global_APE-pos_rmse": self.ape_pos_stats["rmse"],
            f"global_APE-rot_rmse": self.ape_rot_stats["rmse"],
            f"global_APE-deg_rmse": self.ape_deg_stats["rmse"],
            f"global_RPE_rmse": self.rpe_full_stats["rmse"],
            f"global_RPE-pos_rmse": self.rpe_pos_stats["rmse"],
            f"global_RPE-rot_rmse": self.rpe_rot_stats["rmse"],
            f"global_RPE-deg_rmse": self.rpe_deg_stats["rmse"],
        }
        return {
            **{f"global_APE_{k}": v for k, v in self.ape_full_stats.items()},
            **{f"global_APE-pos_{k}": v for k, v in self.ape_pos_stats.items()},
            **{f"global_APE-rot_{k}": v for k, v in self.ape_rot_stats.items()},
            **{f"global_APE-deg_{k}": v for k, v in self.ape_deg_stats.items()},
            **{f"global_RPE_{k}": v for k, v in self.rpe_full_stats.items()},
            **{f"global_RPE-pos_{k}": v for k, v in self.rpe_pos_stats.items()},
            **{f"global_RPE-rot_{k}": v for k, v in self.rpe_rot_stats.items()},
            **{f"global_RPE-deg_{k}": v for k, v in self.rpe_deg_stats.items()},
        }
    
    def get_statistics_str(self, sep="\n"):
        return sep.join([f"{k}: {v:.4f}" for k, v in self.get_statistics().items()])
    
    def print_statistics(self, fn_logger=None):
        if fn_logger is None:
            fn_logger = print
        print("Global Pose Error Metrics:")
        for k, v in self.get_statistics().items():
            fn_logger(f"\t{k}: {v:.4f}")


def align_kf_traj(npz_path, stream, return_full_est_traj=False):
    times, poses = OutputTrajectory.from_video_npz(npz_path).get_trajectory()
    return align_traj(times, poses, stream, return_full_est_traj=return_full_est_traj)


# def align_full_traj(traj_est_full,stream):
#     return align_traj(range(len(traj_est_full)), traj_est_full, stream)


def align_traj(est_timestamps, est_c2ws, stream, return_full_est_traj=False):

    traj_ref = []
    traj_est = []
    timestamps = []

    for i in range(est_timestamps.shape[0]):
        timestamp = int(est_timestamps[i])
        val = stream.poses[timestamp].sum()
        if np.isnan(val) or np.isinf(val):
            print(f'Nan or Inf found in gt poses, skipping {i}th pose!')
            continue
        traj_est.append(est_c2ws[i])  # relative poses (relative to first pose)
        traj_ref.append(stream.poses[timestamp])  # absolute poses
        timestamps.append(est_timestamps[i])

    traj_est =PoseTrajectory3D(poses_se3=traj_est,timestamps=timestamps)
    traj_ref =PoseTrajectory3D(poses_se3=traj_ref,timestamps=timestamps)

    traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)
    r_a, t_a, s = traj_est.align(traj_ref, correct_scale=True)  # alignment removes average absolute pose error and first pose reference position

    if return_full_est_traj:
        traj_est_full = PoseTrajectory3D(poses_se3=est_c2ws, timestamps=est_timestamps)
        traj_est_full.scale(s)
        traj_est_full.transform(lie.se3(r_a, t_a))
        traj_est = traj_est_full

    return r_a, t_a, s, traj_est, traj_ref


def traj_eval_and_plot(traj_est, traj_ref, plot_parent_dir, plot_name):
    if not os.path.exists(plot_parent_dir):
        os.makedirs(plot_parent_dir)

    print("calculating APE+RPE ...")
    pe_evaler = PoseErrorGlobalEvaler()
    pe_evaler.process_data(traj_ref, traj_est)

    print("plotting ...")

    plot_collection = plot.PlotCollection("kf factor graph")

    titles = ["APE-pos", "APE-rot", "RPE-pos", "RPE-rot"]
    metrics = [pe_evaler.ape_pos_metric,
               pe_evaler.ape_rot_metric,
               pe_evaler.rpe_pos_metric,
               pe_evaler.rpe_rot_metric]
    stats = [pe_evaler.ape_pos_stats,
             pe_evaler.ape_rot_stats,
             pe_evaler.rpe_pos_stats,
             pe_evaler.rpe_rot_stats]
    
    fig = plt.figure(figsize=(16,10))

    for i, (title, metric, stat) in enumerate(zip(titles, metrics, stats)):
        plot_mode = plot.PlotMode.xy
        subplot = 221 + i
        ax = plot.prepare_axis(fig, plot_mode, subplot)
        plot.traj(ax, plot_mode, traj_ref, '--', 'gray', 'reference')
        plot.traj_colormap(
            ax, traj_est, metric.error, plot_mode, min_map=stat["min"],
            max_map=stat["max"], title=f"{title} mapped onto trajectory")
    
    plot_collection.add_figure("2d_PE", fig)

    # save plots
    plot_collection.export(f"{plot_parent_dir}/{plot_name}.png", False)

    return pe_evaler


def kf_traj_eval(npz_path, plot_parent_dir,plot_name, stream, wandb_logger):
    r_a, t_a, s, traj_est, traj_ref = align_kf_traj(npz_path, stream)

    offline_video = dict(np.load(npz_path))
    
    if not os.path.exists(plot_parent_dir):
        os.makedirs(plot_parent_dir)

    pe_evaler = traj_eval_and_plot(traj_est,traj_ref,plot_parent_dir,plot_name)

    output_str = "#"*10+"Keyframes traj"+"#"*10+"\n"
    output_str += f"scale: {s}\n"
    output_str += f"rotation:\n{r_a}\n"
    output_str += f"translation:{t_a}\n"
    stats_str = pe_evaler.get_statistics_str(sep='\n\t')
    output_str += f"PE statistics:\n\t{stats_str}\n"
    output_str += "#"*34+"\n"

    print(output_str)
    pe_evaler.print_statistics()
    pe_statistics = pe_evaler.get_statistics()

    out_path=f'{plot_parent_dir}/metrics_kf_traj.txt'
    with open(out_path, 'w+') as fp:
        fp.write(output_str)
    if wandb_logger is not None:
        statistics = {f'kf_{k}':v for k,v in pe_statistics.items()}
        # statistics['pose_scale'] = s
        wandb_logger.log(statistics)

    offline_video["scale"]=np.array(s)
    np.savez(npz_path,**offline_video)

    est_camera_vis = CameraPoseVisualizer()
    est_camera_vis.add_traj(traj_est.poses_se3)
    est_camera_vis.save(f"{plot_parent_dir}/{plot_name}_3d.png")

    ref_camera_vis = CameraPoseVisualizer()
    ref_camera_vis.add_traj(traj_ref.poses_se3)
    ref_camera_vis.save(f"{plot_parent_dir}/ref_3d.png")

    return pe_statistics, s, r_a, t_a


def full_traj_eval(out_traj, plot_parent_dir, plot_name, stream, wandb_logger):
    
    timestamps, traj_est = out_traj.get_trajectory()
    traj_est_not_align = traj_est.copy()
    r_a, t_a, s, traj_est, traj_ref = align_traj(timestamps, traj_est, stream)

    if not os.path.exists(plot_parent_dir):
        os.makedirs(plot_parent_dir)

    pe_evaler = traj_eval_and_plot(traj_est,traj_ref,plot_parent_dir,plot_name)
    
    output_str = "#"*10+"Full traj"+"#"*10+"\n"
    output_str += f"scale: {s}\n"
    output_str += f"rotation:\n{r_a}\n"
    output_str += f"translation:{t_a}\n"
    stats_str = pe_evaler.get_statistics_str(sep='\n\t')
    output_str += f"PE statistics:\n\t{stats_str}\n"
    output_str += "#"*29+"\n"

    print(output_str)
    pe_evaler.print_statistics()

    out_path=f'{plot_parent_dir}/metrics_full_traj.txt'
    with open(out_path, 'w+') as fp:
        fp.write(output_str)
    if wandb_logger is not None:
        statistics = {f'full_{k}':v for k,v in pe_evaler.get_statistics().items()}
        wandb_logger.log(statistics)

    return traj_est_not_align, traj_est, traj_ref


def out_traj_eval(out_traj: OutputTrajectory, plot_parent_dir, plot_name, stream, wandb_logger):

    # align
    timestamps, traj_est = out_traj.get_trajectory()
    traj_est_not_align = traj_est.copy()
    r_a, t_a, s, traj_est, traj_ref = align_traj(timestamps, traj_est, stream)

    # evaluate and plot
    if not os.path.exists(plot_parent_dir):
        os.makedirs(plot_parent_dir)

    pe_evaler = traj_eval_and_plot(traj_est,traj_ref,plot_parent_dir,plot_name)
    
    # show statistics
    output_str = "#"*10+"Out traj"+"#"*10+"\n"
    output_str += f"scale: {s}\n"
    output_str += f"rotation:\n{r_a}\n"
    output_str += f"translation:{t_a}\n"
    stats_str = pe_evaler.get_statistics_str(sep='\n\t')
    output_str += f"PE statistics:\n\t{stats_str}\n"
    output_str += "#"*29+"\n"

    print(output_str)
    pe_evaler.print_statistics()

    out_path=f'{plot_parent_dir}/metrics_out_traj.txt'
    with open(out_path, 'w+') as fp:
        fp.write(output_str)

    if wandb_logger is not None:
        statistics = {f'out_{k}':v for k,v in pe_evaler.get_statistics().items()}
        wandb_logger.log(statistics)

    return traj_est_not_align, traj_est, traj_ref  # TODO return scale too
