import modeling
import optimization
import tokenization
import tensorflow as tf
import uuid


class InputExample(object):
    """简单的序列分类中的单个训练/测试样本"""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """
          guid: 标记样本的唯一性id
          text_a: string. 第一个untokenized的文本序列，对于单个文本序列的任务，只需要指定这一个序列
          text_b: (Optional) string. 第二个untokenized的文本序列， 对于文本对的任务，需要指定第二个文本序列
          label: (Optional) string. 单个样本的标签，对于训练和验证样本，必须要指定该参数，但对测试样本不需要
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, is_real_example=True):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.is_real_example = is_real_example


class BertSentiment:

    def __init__(self):
        # 配置信息
        self.bert_config_file = "models/chinese_L-12_H-768_A-12/bert_config.json"
        self.vocab_file = "models/chinese_L-12_H-768_A-12/vocab.txt"
        self.output_dir = "data_output_128"
        self.init_checkpoint = "data_output_sentiment_128/model.ckpt-5999"
        self.label_list = ["0", "1"]   # 0表示不好，1表示好
        self.do_lower_case = True
        self.max_seq_length = 128
        self.train_batch_size = 32
        self.eval_batch_size = 8
        self.predict_batch_size = 8
        self.learning_rate = 5e-5
        self.save_checkpoints_steps = 1000
        self.iterations_per_loop = 1000
        self.use_tpu = False
        self.master = None
        self.num_tpu_cores = 8
        self.gpu_memory_fraction = 0.6  # 设置GPU显存占有率
        self._load_model()

    @staticmethod
    def example_to_feature(example, label_list, max_seq_length, tokenizer):
        """将单个InputExample对象转换为单个InputFeatures对象"""

        label_map = {}
        for i, label in enumerate(label_list):
            label_map[label] = i

        # 分词（按单个汉字来分）
        tokens_a = tokenizer.tokenize(example.text_a)
        # 例如tokenizer.tokenize("接到上级指示 我负责你们在中国的安全[嘻嘻]")的结果为
        # ['接', '到', '上', '级', '指', '示', '我', '负', '责', '你', '们', '在', '中', '国', '的', '安', '全', '[', '嘻', '嘻', ']']

        # 如果字符数量（包括[CLS]和[SEP]）超过max_seq_length，则进行截断操作
        if len(tokens_a) + 2 > max_seq_length:
            tokens_a = tokens_a[0:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0     0   0   0  0     0 0

        # tokens = []
        # segment_ids = []
        # tokens.append("[CLS]")
        # segment_ids.append(0)
        # for token in tokens_a:
        #     tokens.append(token)
        #     segment_ids.append(0)
        # tokens.append("[SEP]")
        # segment_ids.append(0)

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * (len(tokens_a) + 2)  # 用来标记所属的句子，如果属于第1个句子，则值为0，如果属于第2个句子，则值为1，以此类推

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        # 例如结果为[101, 2970, 1168, 677, 5277, 2900, ...., 1744, 4638, 2128, 1059, 138, 1677, 1677, 140, 102]
        # 表示分词得到的汉字在字典文件中的编号
        input_mask = [1] * len(input_ids)  # 掩码中的1表示实际的token，0表示填充的token，

        # 进行零填充
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]

        feature = InputFeatures(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids, label_id=label_id,
                                is_real_example=True)
        return feature

    @staticmethod
    def examples_to_features(examples, label_list, max_seq_length, tokenizer):
        """将InputExample对象的列表 转化为 InputFeatures对象的列表 """
        features = []
        for ex_index, example in enumerate(examples):
            feature = BertSentiment.example_to_feature(example, label_list, max_seq_length, tokenizer)
            features.append(feature)
        return features

    @staticmethod
    def input_fn_builder(features, seq_length, is_training, drop_remainder):
        """Creates an `input_fn` closure to be passed to TPUEstimator."""

        all_input_ids = []
        all_input_mask = []
        all_segment_ids = []
        all_label_ids = []

        for feature in features:
            all_input_ids.append(feature.input_ids)
            all_input_mask.append(feature.input_mask)
            all_segment_ids.append(feature.segment_ids)
            all_label_ids.append(feature.label_id)

        def input_fn(params):
            """The actual input function."""
            batch_size = params["batch_size"]

            num_examples = len(features)

            # This is for demo purposes and does NOT scale to large data sets. We do
            # not use Dataset.from_generator() because that uses tf.py_func which is
            # not TPU compatible. The right way to load data is with TFRecordReader.
            d = tf.data.Dataset.from_tensor_slices({
                "input_ids": tf.constant(all_input_ids, shape=[num_examples, seq_length], dtype=tf.int32),
                "input_mask": tf.constant(all_input_mask, shape=[num_examples, seq_length], dtype=tf.int32),
                "segment_ids": tf.constant(all_segment_ids, shape=[num_examples, seq_length], dtype=tf.int32),
                "label_ids": tf.constant(all_label_ids, shape=[num_examples], dtype=tf.int32),
            })

            if is_training:
                d = d.repeat()
                d = d.shuffle(buffer_size=100)

            d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
            """
            d的结果为<DatasetV1Adapter 
            shapes: {input_ids: (?, 128), input_mask: (?, 128), segment_ids: (?, 128), label_ids: (?,)}, 
            types: {input_ids: tf.int32, input_mask: tf.int32, segment_ids: tf.int32, label_ids: tf.int32}>
            """
            return d

        return input_fn

    @staticmethod
    def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                     labels, num_labels, use_one_hot_embeddings, id):
        """Creates a classification model."""
        model = modeling.BertModel(
            config=bert_config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=use_one_hot_embeddings)

        # In the demo, we are doing a simple classification task on the entire
        # segment.
        #
        # If you want to use the token-level output, use model.get_sequence_output()
        # instead.
        output_layer = model.get_pooled_output()

        hidden_size = output_layer.shape[-1].value

        output_weights = tf.get_variable(
            "output_weights", [num_labels, hidden_size],
            initializer=tf.truncated_normal_initializer(stddev=0.02))

        output_bias = tf.get_variable(
            "output_bias", [num_labels], initializer=tf.zeros_initializer())

        with tf.variable_scope("loss"):
            if is_training:
                # I.e., 0.1 dropout
                output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

            logits = tf.matmul(output_layer, output_weights, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)
            probabilities = tf.nn.softmax(logits, axis=-1)
            log_probs = tf.nn.log_softmax(logits, axis=-1)

            one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

            per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
            loss = tf.reduce_mean(per_example_loss)

            return (loss, per_example_loss, logits, probabilities, id)

    @staticmethod
    def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                         num_train_steps, num_warmup_steps, use_tpu, use_one_hot_embeddings):
        """Returns `model_fn` closure for TPUEstimator."""

        def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
            """The `model_fn` for TPUEstimator."""

            tf.logging.info("*** Features ***")
            for name in sorted(features.keys()):
                tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

            input_ids = features["input_ids"]
            input_mask = features["input_mask"]
            segment_ids = features["segment_ids"]
            label_ids = features["label_ids"]
            id = features['id']
            is_real_example = None
            if "is_real_example" in features:
                is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
            else:
                is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

            is_training = (mode == tf.estimator.ModeKeys.TRAIN)

            (total_loss, per_example_loss, logits, probabilities, id) = BertSentiment.create_model(
                bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
                num_labels, use_one_hot_embeddings, id)

            tvars = tf.trainable_variables()
            initialized_variable_names = {}
            scaffold_fn = None
            if init_checkpoint:
                (assignment_map, initialized_variable_names
                 ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
                if use_tpu:
                    def tpu_scaffold():
                        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                        return tf.train.Scaffold()

                    scaffold_fn = tpu_scaffold
                else:
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

            tf.logging.info("**** Trainable Variables ****")
            for var in tvars:
                init_string = ""
                if var.name in initialized_variable_names:
                    init_string = ", *INIT_FROM_CKPT*"
                tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape, init_string)

            output_spec = None
            if mode == tf.estimator.ModeKeys.TRAIN:

                train_op = optimization.create_optimizer(
                    total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

                output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                    mode=mode,
                    loss=total_loss,
                    train_op=train_op,
                    scaffold_fn=scaffold_fn)
            elif mode == tf.estimator.ModeKeys.EVAL:

                def metric_fn(per_example_loss, label_ids, logits, is_real_example):
                    predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
                    accuracy = tf.metrics.accuracy(
                        labels=label_ids, predictions=predictions, weights=is_real_example)
                    loss = tf.metrics.mean(values=per_example_loss, weights=is_real_example)
                    return {
                        "eval_accuracy": accuracy,
                        "eval_loss": loss,
                    }

                eval_metrics = (metric_fn,
                                [per_example_loss, label_ids, logits, is_real_example])
                output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                    mode=mode,
                    loss=total_loss,
                    eval_metrics=eval_metrics,
                    scaffold_fn=scaffold_fn)
            else:
                output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                    mode=mode,
                    predictions={'id': id, "probabilities": probabilities},
                    scaffold_fn=scaffold_fn)
            return output_spec

        return model_fn

    def _load_model(self):
        # 加载模型超参数配置信息
        bert_config = modeling.BertConfig.from_json_file(self.bert_config_file)

        # 检查max_seq_length参数是否超过当前预训练模型的最大限制
        if self.max_seq_length > bert_config.max_position_embeddings:
            raise ValueError("sequence length cann't exceed the maximum limit")

        # 配置GPU使用策略
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        # config.gpu_options.per_process_gpu_memory_fraction = self.gpu_memory_fraction

        # 创建配置信息对象
        is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
        run_config = tf.contrib.tpu.RunConfig(cluster=None, master=self.master,
                                              # model_dir=self.output_dir,
                                              save_checkpoints_steps=self.save_checkpoints_steps,
                                              session_config=config,
                                              tpu_config=tf.contrib.tpu.TPUConfig(
                                                  iterations_per_loop=self.iterations_per_loop,
                                                  num_shards=self.num_tpu_cores,
                                                  per_host_input_for_training=is_per_host))

        # 创建模型函数的闭包
        model_fn = BertSentiment.model_fn_builder(bert_config=bert_config, num_labels=len(self.label_list),
                                                  init_checkpoint=self.init_checkpoint,
                                                  learning_rate=self.learning_rate,
                                                  num_train_steps=None, num_warmup_steps=None, use_tpu=self.use_tpu,
                                                  use_one_hot_embeddings=self.use_tpu)

        # 创建估算器estimator进行训练，如果TPU不可用，则它会自动回退到正常的基于CPU或GPU的Estimator
        self.estimator = tf.contrib.tpu.TPUEstimator(use_tpu=self.use_tpu, model_fn=model_fn, config=run_config,
                                                     train_batch_size=self.train_batch_size,
                                                     eval_batch_size=self.eval_batch_size,
                                                     predict_batch_size=self.predict_batch_size)
        # 加载分词器
        self.tokenizer = tokenization.FullTokenizer(vocab_file=self.vocab_file, do_lower_case=self.do_lower_case)

    def input_fn_builder2(self):
        def gen():
            while True:
                text = input("please input a sentence:")  # 一次从redis中读取一个文本并yield一个文本，包含文本中的id
                # guid这里实际用不到，可以随便写，但是label必须是label_list中的一个
                examples = [InputExample(guid=0, text_a=text, text_b=None, label="0")]
                features = BertSentiment.examples_to_features(examples, self.label_list, self.max_seq_length,
                                                              self.tokenizer)

                all_input_ids = []
                all_input_mask = []
                all_segment_ids = []
                all_label_ids = []

                for feature in features:
                    all_input_ids.append(feature.input_ids)
                    all_input_mask.append(feature.input_mask)
                    all_segment_ids.append(feature.segment_ids)
                    all_label_ids.append(feature.label_id)

                # 客户端的唯一性标记，在create_model()方法的参数和返回值中都加了id才行，model_fn方法里也要加上id：
                # predictions={'id': id, "probabilities": probabilities}
                yield {'id': str(uuid.uuid4()),
                       'input_ids': all_input_ids,
                       'input_mask': all_input_mask,
                       'segment_ids': all_segment_ids,
                       'label_ids': all_label_ids,
                       }

        def input_fn(params):
            # batch_size = params["batch_size"]
            types = {'id': tf.string,
                     'input_ids': tf.int32,
                     'input_mask': tf.int32,
                     'segment_ids': tf.int32,
                     'label_ids': tf.int32,
                     }
            shapes = {'id': (),
                      'input_ids': (None, self.max_seq_length),
                      'input_mask': (None, self.max_seq_length),
                      'segment_ids': (None, self.max_seq_length),
                      'label_ids': (None,),
                      }
            return tf.data.Dataset.from_generator(gen, output_types=types, output_shapes=shapes).prefetch(1)

        return input_fn

    def predict(self):
        """ 预测文本列表中批量文本的情感极性 """

        for result in self.estimator.predict(self.input_fn_builder2(), yield_single_examples=False):
            print("raw result:", result)
            rst = "positive" if result['probabilities'][0][0] <= result['probabilities'][0][1] else "negative"
            print("the sentence you input is {}".format(rst))


if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'

    BertSentiment().predict()

