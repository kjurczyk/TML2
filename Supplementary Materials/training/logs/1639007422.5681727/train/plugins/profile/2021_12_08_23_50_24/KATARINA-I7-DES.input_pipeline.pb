		?c?#@	?c?#@!	?c?#@	2sl?]F@2sl?]F@!2sl?]F@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$	?c?#@ۊ?e????A?s?2@Y?q???@*	???????@2P
Iterator::Model::Prefetch:??H?@!K??`O{G@):??H?@1K??`O{G@:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMap?{??P+@!??b3J@)??"??~@15@K?A??@:Preprocessing2g
0Iterator::Model::Prefetch::FlatMap[0]::Generator?7??d???!*?h??14@)?7??d???1*?h??14@:Preprocessing2F
Iterator::ModelX?5?;?@!kk????G@)???ZӼ??1??8UO???:Preprocessing2e
.Iterator::Model::Prefetch::FlatMap::FromTensora2U0*?S?!?YnʊĊ?)a2U0*?S?1?YnʊĊ?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 44.7% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*no92sl?]F@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	ۊ?e????ۊ?e????!ۊ?e????      ??!       "      ??!       *      ??!       2	?s?2@?s?2@!?s?2@:      ??!       B      ??!       J	?q???@?q???@!?q???@R      ??!       Z	?q???@?q???@!?q???@JCPU_ONLYY2sl?]F@b 