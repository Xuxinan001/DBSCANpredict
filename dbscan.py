import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
import json
from sklearn.metrics.pairwise import haversine_distances
import plotly.graph_objects as go


# def find_closest_node(target_node, node_list):
#     closest_node_index = -1
#     minimum_distance = 1e12
#
#     for index, current_node in enumerate(node_list):
#         distance_to_current_node = (
#                 haversine_distances(np.radians([target_node]), np.radians([current_node]))[0][0] * 6371
#         )
#
#         if distance_to_current_node < minimum_distance:
#             minimum_distance = distance_to_current_node
#             closest_node_index = index
#
#     if minimum_distance < 500 :
#         closest_node_index=closest_node_index
#         print(minimum_distance)
#     else :
#         closest_node_index=-1
#     # print(closest_node_index)
#     return  closest_node_index


# 加载数据
data = pd.read_csv('new_lng.csv', header=None)
data.columns = ['mmsi', '时间', '航行状态', '速度', '经度', '纬度', '吃水']
# 进行随机抽样
sample_fraction = 0.01  # 抽样0.1%的数据
data = data.sample(frac=sample_fraction)

# 选择经度和纬度坐标
coordinates = data[['经度', '纬度']]


# 应用DBSCAN聚类
db = DBSCAN(eps=1/500, min_samples=100).fit(coordinates)
labels = db.labels_

# 初始化存储聚类中心和大小的列表
cluster_centers, cluster_sizes = [], []

# 对每个唯一标签（聚类），计算其中心和大小
unique_labels = set(labels)
for label in unique_labels:
    if label != -1:
        cluster_samples = coordinates[labels == label]
        cluster_center = cluster_samples.mean(axis=0)
        cluster_centers.append(cluster_center)
        cluster_sizes.append(len(cluster_samples))


def analyze_cluster_centers(data, cluster_centers):
    RADIUS_KM = 30  # 考虑的半径，单位：千米
    cluster_status = [0 for _ in range(len(cluster_centers))]

    for index, center in enumerate(cluster_centers):
        # 计算所有数据点到当前聚类中心的距离
        data['distance_to_center'] = haversine_distances(
            np.radians([center]),
            np.radians(data[['经度', '纬度']].values)
        ).flatten() * 6371  # 转换为千米，将二维数组转化为一维数组
        # print(center)
        print(data.sort_values('distance_to_center'))
        # 筛选出距离聚类中心10km内的数据点
        close_data = data[data['distance_to_center'] <= RADIUS_KM]

        # 按照mmsi（船的标识）和时间对close_data进行排序
        close_data_sorted = close_data.sort_values(['mmsi', '时间'])
        # print(close_data_sorted)
        previous_row = None
        is_first_row = True

        for current_row in close_data_sorted.itertuples(index=False):
            if is_first_row:
                is_first_row = False
            elif abs(current_row.吃水 - previous_row.吃水) > 1 and current_row.mmsi == previous_row.mmsi:
                cluster_status[index] += 1 if current_row.吃水 > previous_row.吃水 else -1
            previous_row = current_row

        # 删除distance_to_center列，以节省内存
        del data['distance_to_center']

    return cluster_status





def write_output_file(cluster_centers, cluster_status):
    """Write output to a JSON file."""
    with open("lng.json", "w") as file:
        lng_entry_code = 1
        file.write("[\n")

        # Write LNG true records
        for index, center in enumerate(cluster_centers):
            if abs(cluster_status[index]) > 2:
                print(cluster_status[index])
                record = {
                    "code": lng_entry_code,
                    "latitude": center[0],
                    "longtitude": center[1],
                    "isLNG": True,
                    "IN": cluster_status[index] < 0
                }
                json.dump(record, file)
                file.write(",\n")
                lng_entry_code += 1

        # Write LNG false records
        for index, center in enumerate(cluster_centers):
            if abs(cluster_status[index]) <= 2:
                print(cluster_status[index])
                record = {
                    "code": lng_entry_code,
                    "latitude": center[0],
                    "longtitude": center[1],
                    "isLNG": False,
                    "IN": None
                }
                json.dump(record, file)
                if index != len(cluster_centers) - 1:
                    file.write(",\n")
                lng_entry_code += 1

        file.write("\n]")


# Use the functions
cluster_status = analyze_cluster_centers(data, cluster_centers)
write_output_file(cluster_centers, cluster_status)
print("----")
print(cluster_status)
# 为每个聚类中心创建一个散点
scatter = go.Scattergeo(
    lon=[center['经度'] for center in cluster_centers],
    lat=[center['纬度'] for center in cluster_centers],
    text=[f'Cluster {i + 1}, samples: {size}' for i, size in enumerate(cluster_sizes)],
    mode='markers',
    marker=dict(
        size=10,
        opacity=0.8,
    ))

# 添加散点到地图
data = [scatter]

# 设置地图的布局
layout = dict(
    title='LNG Stations',
    geo=dict(showland=True)
)

# 创建地图对象并显示
fig = go.Figure(data=data, layout=layout)
fig.show()
