	???(\?@???(\?@!???(\?@	???ЦUM@???ЦUM@!???ЦUM@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$???(\?@鷯???ARI??&B@Y?|гYu@*	33333$?@2P
Iterator::Model::Prefetch^K?=?@!???3?I@)^K?=?@1???3?I@:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMapt$???~@!???Q<?G@)#J{?/L@1?Ț??C@:Preprocessing2g
0Iterator::Model::Prefetch::FlatMap[0]::GeneratorRI??&???!t??d!@)RI??&???1t??d!@:Preprocessing2F
Iterator::Model???V?/@!* *?? J@)????镲?1?Y?^)$??:Preprocessing2e
.Iterator::Model::Prefetch::FlatMap::FromTensor??H?}M?!>??Q???)??H?}M?1>??Q???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 58.7% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*no9???ЦUM@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	鷯???鷯???!鷯???      ??!       "      ??!       *      ??!       2	RI??&B@RI??&B@!RI??&B@:      ??!       B      ??!       J	?|гYu@?|гYu@!?|гYu@R      ??!       Z	?|гYu@?|гYu@!?|гYu@JCPU_ONLYY???ЦUM@b 