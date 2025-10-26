import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

# gt_path = r"/data_C/minzhi/datasets/MM_Classification/test_1k/test_labels.txt"  # 主办方专有
# sub_path = r"logs/2/submission.csv"
#
# gt = pd.read_csv(gt_path, sep=r'\s+', header=None, names=['filename', 'label_gt'])
# sub = pd.read_csv(sub_path)
#
# # 文件名匹配合并，确保只比较出现在两边的数据
# merged = gt.merge(sub, on='filename', how='inner', suffixes=('_gt', '_pred'))
#
# acc = accuracy_score(merged['label_gt'], merged['label_pred'])
# print(f"总体准确率: {acc*100:.2f}%")
#
# # 调用 sklearn.metrics.classification_report() , 输出每个类别的多种统计指标
# # macro avg:	所有类的平均值（不考虑样本数量）
# # weighted avg:	各类指标按样本数量加权平均
# print(classification_report(merged['label_gt'], merged['label_pred'], digits=4))
#
#
# # # coding=utf-8
# # import sys
# # # 用于打印JSON编码的评分结果
# # import json
# #
# # def eval(submit_file):
# #     """ 评分函数
# #         :param submit_file: 选手提交文件
# #         :return: dict:分数结果
# #     """
# #     # TODO 1.提交文件校验
# #     file_name = submit_file[1]
# #     with open(file_name) as f:
# #         score_str = f.readline()
# #     # TODO 2.具体评分逻辑
# #
# #     # 正确分类信息文件，该文件随本评测脚本一起压缩上传
# #     groundTruth = "groundTruth.csv"
# #
# #     # TODO 3.返回类型为Dict, key不支持修改与增减
# #     return {
# #         "score": 92.5,              #替换value为最终评测分数
# #         "errorMsg": "success",      #错误提示信息，仅在code值为非0时打印
# #         "code": 0,                  #code值为0打印score，非0打印errorMsg
# #         "data": [
# #             {
# #                 "score": int(score_str)
# #             }
# #         ]
# #     }
# #
# # if __name__ == '__main__':
# #     # 打印格式必须为JSON编码的字符串
# #     print(json.dumps(eval(sys.argv)))
#
# coding=utf-8
import sys
import json
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

def eval(submit_file):
    """评测函数
    :param submit_file: [脚本名, 提交文件路径]
    :return: dict 格式的评测结果（平台要求的格式）
    """

    # --------------------------
    # Step 1. 读取提交文件路径
    # --------------------------
    try:
        submit_path = submit_file[1]  # [1]  # 平台传入 [脚本名, 提交路径]
    except IndexError:
        return {
            "score": 0.0,
            "errorMsg": "缺少提交文件",
            "code": 1,
            "data": []
        }

    # --------------------------
    # Step 2. 加载 ground truth 与 submission
    # --------------------------
    try:
        # 正确标签文件（平台随脚本打包上传）
        gt_path = "groundTruth.txt"  # MM_Classification/test_1k/test_labels.txt"  # "groundTruth.csv" 或 groundTruth.txt
        gt = pd.read_csv(gt_path, sep=r'\s+', header=None, names=['filename', 'label_gt'])

        # 选手提交文件
        sub = pd.read_csv(submit_path)
        if not {'filename', 'label_pred'}.issubset(sub.columns):
            raise ValueError("提交文件缺少必须列：filename 或 label_pred")

    except Exception as e:
        return {
            "score": 0.0,
            "errorMsg": f"文件读取错误: {str(e)}",
            "code": 2,
            "data": []
        }

    # --------------------------
    # Step 3. 匹配并计算分数
    # --------------------------
    try:
        # 按 filename 匹配
        merged = gt.merge(sub, on='filename', how='inner', suffixes=('_gt', '_pred'))

        if len(merged) == 0:
            raise ValueError("文件名无匹配项，可能文件名格式不符")

        acc = accuracy_score(merged['label_gt'], merged['label_pred'])
        report = classification_report(
            merged['label_gt'], merged['label_pred'], digits=4, output_dict=True
        )

        macro_avg = report['macro avg']['f1-score']
        weighted_avg = report['weighted avg']['f1-score']

        # 综合评分（可按任务规则调整）
        final_score = acc * 100

    except Exception as e:
        return {
            "score": 0.0,
            "errorMsg": f"评分计算失败: {str(e)}",
            "code": 3,
            "data": []
        }

    # --------------------------
    # Step 4. 返回 JSON 结果
    # --------------------------
    return {
        "score": round(final_score, 4),     # 最终得分（百分比）
        "errorMsg": "success",
        "code": 0,                          # 0 表示正常评测
        "data": [
            {
                "accuracy": round(acc, 4),
                "macro_f1": round(macro_avg, 4),
                "weighted_f1": round(weighted_avg, 4),
                "total_samples": int(len(merged))
            }
        ]
    }


if __name__ == '__main__':
    # 打印 JSON 编码的结果字符串（平台要求）r"logs/2/submission.csv"
    print(json.dumps(eval(sys.argv)))
