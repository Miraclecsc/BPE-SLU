import random
import json
import copy

with open('./data/train.json', 'r', encoding='utf-8') as file:
    origin = json.load(file)
# with open('./data/error_data.json', 'r', encoding='utf-8') as file:
#     origin = json.load(file)

# 这些slots的value都在poi_name文件中
poi_slots = ['poi名称', 'poi修饰', 'poi目标', '起点名称', '起点修饰', '起点目标',
             '终点名称', '终点修饰', '终点目标', '途经点名称']
poi_values = [c.rstrip()
              for c in open('./data/lexicon/poi_name.txt', encoding='utf-8')]


# 请求类型
request_values = ["附近", "定位", "近郊", "旁边", "周边", "就近", "最近"]

#  路线偏好
preferenece_values = ["最近", "高速优先", "走国道", "少走高速", "不走高速", "走高速",
                      "上高速", "高速公路", "最快", "躲避拥堵"]

# 对象
object_values = ["语音", "高德地图", "路线", "位置", "途经点", "全程路线",
                 "简易导航", "目的地", "地图", "定位", "路况", "导航"]

appendix = []

for idx in range(len(origin)):
    appendix.append(origin[idx])
    for item_id in range(len(origin[idx])):

        item = origin[idx][item_id]
        # item["utt_id"] = 1 # utt_id of new items are all 1

        manual_transcript = origin[idx][item_id]['manual_transcript']
        semantic = origin[idx][item_id]['semantic']
        # print(manual_transcript)
        # print(semantic)

        for semantic_idx in range(len(semantic)):
            slot = semantic[semantic_idx][1]
            value = semantic[semantic_idx][2]
            # print(slot, value)
            # assert 0

            if value not in manual_transcript:
                continue

            if slot in poi_slots:
                for _ in range(2):
                    v = random.choice(poi_values)

                    tmp = copy.deepcopy(item)
                    tmp["utt_id"] = 1
                    tmp["semantic"][semantic_idx][2] = v
                    tmp['manual_transcript'] = manual_transcript.replace(
                        value, v)
                    appendix.append([tmp])

            if slot == "请求类型":
                for v in request_values:
                    tmp = copy.deepcopy(item)
                    tmp["utt_id"] = 1
                    tmp["semantic"][semantic_idx][2] = v
                    tmp['manual_transcript'] = manual_transcript.replace(
                        value, v)
                    appendix.append([tmp])

            if slot == "路线偏好":
                for v in preferenece_values:
                    tmp = copy.deepcopy(item)
                    tmp["utt_id"] = 1
                    tmp["semantic"][semantic_idx][2] = v
                    tmp['manual_transcript'] = manual_transcript.replace(
                        value, v)
                    appendix.append([tmp])
            if slot == "对象":
                for v in object_values:
                    tmp = copy.deepcopy(item)
                    tmp["utt_id"] = 1
                    tmp["semantic"][semantic_idx][2] = v
                    tmp['manual_transcript'] = manual_transcript.replace(
                        value, v)
                    appendix.append([tmp])

print(len(appendix))
# print(appendix[0:10])
augment = json.dumps(appendix, indent=4, ensure_ascii=False)

with open('./data/train_augment.json', 'w') as wf:
    print(augment, file=wf)
