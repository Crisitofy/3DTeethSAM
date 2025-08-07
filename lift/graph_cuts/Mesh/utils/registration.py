import sys
sys.path.append("..")
import os


_cur_path = os.path.dirname(os.path.realpath(__file__))

def tooth_registration(path_standard, mesh_other):
    import open3d as o3d
    # 读取文件
    mesh_standard = o3d.io.read_triangle_mesh(path_standard)
    # mesh_other = o3d.io.read_triangle_mesh(path_other)

    # 经验参数
    voxel_size = 2.0
    # 化简
    pcd_standard = mesh_standard.sample_points_uniformly(5000)
    pcd_other = mesh_other.sample_points_uniformly(5000)
    # pcd_standard = o3d.voxel_down_sample(pcd_standard, voxel_size=voxel_size)
    # pcd_other = o3d.voxel_down_sample(pcd_other, voxel_size=voxel_size)

    # color
    # pcd_standard.paint_uniform_color([1, 0, 0])
    # pcd_other.paint_uniform_color([0, 1, 0])
    # 经验参数
    radius_normal = voxel_size * 2

    pcd_standard.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=15))
    pcd_other.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=15))

    # 经验参数
    radius_feature = voxel_size * 3
    pcd_standard_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd_standard,
                                                              o3d.geometry.KDTreeSearchParamHybrid(
                                                                  radius=radius_feature,
                                                                  max_nn=15))
    pcd_other_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd_other,
                                                           o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature,
                                                                                                max_nn=15))
    # 经验参数
    distance_threshold = voxel_size

    # global registration
    result_g_1 = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        pcd_other, pcd_standard, pcd_other_fpfh, pcd_standard_fpfh, True, distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.6),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 1000))

    result_g_2 = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        pcd_other, pcd_standard, pcd_other_fpfh, pcd_standard_fpfh, True, distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.6),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 1000))
    if result_g_1.transformation[0][0] * result_g_2.transformation[0][0] < 0:
        result_g_3 = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            pcd_other, pcd_standard, pcd_other_fpfh, pcd_standard_fpfh, True, distance_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3, [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.6),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                    distance_threshold)
            ], o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 1000))
        result_g_4 = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            pcd_other, pcd_standard, pcd_other_fpfh, pcd_standard_fpfh, True, distance_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3, [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.6),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                    distance_threshold)
            ], o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 1000))
        result_g_5 = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            pcd_other, pcd_standard, pcd_other_fpfh, pcd_standard_fpfh, True, distance_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3, [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.6),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                    distance_threshold)
            ], o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 1000))

        count_1 = 0
        count_2 = 0

        if result_g_1.transformation[0][0] * result_g_3.transformation[0][0] >= 0:
            count_1 += 1
        else:
            count_2 += 1

        if result_g_1.transformation[0][0] * result_g_4.transformation[0][0] >= 0:
            count_1 += 1
        else:
            count_2 += 1

        if result_g_1.transformation[0][0] * result_g_5.transformation[0][0] >= 0:
            count_1 += 1
        else:
            count_2 += 1

        if count_1 > count_2:
            result_g = result_g_1
        else:
            result_g = result_g_2


    else:
        result_g = result_g_1

    distance_threshold_icp = voxel_size * 2.0
    # icp
    result_t = 0
    for _ in range(11):
        result = o3d.pipelines.registration.registration_icp(
            pcd_other, pcd_standard, distance_threshold_icp, result_g.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))
        result_t += result.transformation
    result_t /= 11

    threshold_evaluation = 2
    evaluation = o3d.pipelines.registration.evaluate_registration(pcd_other, pcd_standard,
                                                        threshold_evaluation, result_t)

    return result_t, evaluation.fitness

