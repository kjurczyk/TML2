	@a?ӫ@@a?ӫ@!@a?ӫ@	?įT
@?įT
@!?įT
@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$@a?ӫ@?c?ZB??A?D??t	@Ym????ҽ?*	33333??@2g
0Iterator::Model::Prefetch::FlatMap[0]::Generator??MbX??!?b??iW@)??MbX??1?b??iW@:Preprocessing2F
Iterator::Model???<,Ԫ?!vD???@)'???????1??ځ?@:Preprocessing2P
Iterator::Model::Prefetcha2U0*???!???P????)a2U0*???1???P????:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMap??Q????!??!q?W@)?I+?v?1?I?\/J??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 8.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?įT
@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?c?ZB???c?ZB??!?c?ZB??      ??!       "      ??!       *      ??!       2	?D??t	@?D??t	@!?D??t	@:      ??!       B      ??!       J	m????ҽ?m????ҽ?!m????ҽ?R      ??!       Z	m????ҽ?m????ҽ?!m????ҽ?JCPU_ONLYY?įT
@b 