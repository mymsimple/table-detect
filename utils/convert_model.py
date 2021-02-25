# coding: utf-8
"""
模型转换 h5 > pb
"""

import tensorflow as tf
from nets.unet import unet as Unet
import os

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


# 转换函数
def h5_to_pb(h5_model, output_path, out_prefix="output_", input_prefix="input_"):
    saveModDir = output_path

    # with tf.Session() as sess:
    #     tf.global_variables_initializer().run()

    input_nodes = {}
    out_nodes = {}
    input_nodes[input_prefix + "1"] = tf.saved_model.utils.build_tensor_info(h5_model.input)
    out_nodes[out_prefix + "1"] = tf.saved_model.utils.build_tensor_info(h5_model.output)

    # 保存转换训练好的模型
    builder = tf.saved_model.builder.SavedModelBuilder(saveModDir)

    prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(
        inputs=input_nodes,
        outputs=out_nodes,
        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
    )

    builder.add_meta_graph_and_variables(
        sess=tf.keras.backend.get_session(),
        tags=[tf.saved_model.tag_constants.SERVING],
        signature_def_map={  # 保存模型的方法名，与客户端的request.model_spec.signature_name对应
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: prediction_signature
        }
    )
    builder.save()
    print("转换模型结束", saveModDir)



def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--weight_path', default='models/table-line-fine.h5')
    parser.add_argument('-o', '--output_path', default='models/table-line/100000')
    args = parser.parse_args()
    weight_path = args.weight_path
    output_path = args.output_path

    # # 加载网络模型
    model = Unet((1024, 1024, 3), 2)
    # 加载权重模型
    model.load_weights(weight_path)

    # model.summary()

    h5_to_pb(model, output_path)

    print('model saved')


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = "3"
    main()
