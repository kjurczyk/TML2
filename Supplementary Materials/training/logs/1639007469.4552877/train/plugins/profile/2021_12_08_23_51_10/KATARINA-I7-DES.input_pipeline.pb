	&䃞?*"@&䃞?*"@!&䃞?*"@	y?r?E@y?r?E@!y?r?E@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$&䃞?*"@=?U????A
h"lx?@Y?q???@*	fffff.?@2P
Iterator::Model::Prefetch??S??@!_????wG@)??S??@1_????wG@:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMap?C???@@!c???1J@)?|гYu@1hrw?E?@@:Preprocessing2g
0Iterator::Model::Prefetch::FlatMap[0]::Generatorh??s???!?_+2@)h??s???1?_+2@:Preprocessing2F
Iterator::Modelڬ?\m?@!?q*??G@)b??4?8??1?'?$???:Preprocessing2e
.Iterator::Model::Prefetch::FlatMap::FromTensor-C??6J?!?<?P?ƃ?)-C??6J?1?<?P?ƃ?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 43.8% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*no9y?r?E@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	=?U????=?U????!=?U????      ??!       "      ??!       *      ??!       2	
h"lx?@
h"lx?@!
h"lx?@:      ??!       B      ??!       J	?q???@?q???@!?q???@R      ??!       Z	?q???@?q???@!?q???@JCPU_ONLYYy?r?E@b 