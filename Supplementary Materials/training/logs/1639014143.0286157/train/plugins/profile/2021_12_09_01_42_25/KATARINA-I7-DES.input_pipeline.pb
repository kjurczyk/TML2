	????Ƌ@????Ƌ@!????Ƌ@	?$????Q@?$????Q@!?$????Q@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$????Ƌ@Έ?????A??<,Ԛ??Y?W?2?1@*	fffffֻ@2P
Iterator::Model::Prefetch???ׁs@!?*NH??H@)???ׁs@1?*NH??H@:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMap????@!'?KU~wH@)??????@1fNR?_C@:Preprocessing2g
0Iterator::Model::Prefetch::FlatMap[0]::Generatorc?=yX??!???Z%@)c?=yX??1???Z%@:Preprocessing2F
Iterator::Model[Ӽ?@!?y????I@)Q?|a2??1r??L???:Preprocessing2e
.Iterator::Model::Prefetch::FlatMap::FromTensor??H?}M?!Q1??H݉?)??H?}M?1Q1??H݉?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 71.0% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*no9?$????Q@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	Έ?????Έ?????!Έ?????      ??!       "      ??!       *      ??!       2	??<,Ԛ????<,Ԛ??!??<,Ԛ??:      ??!       B      ??!       J	?W?2?1@?W?2?1@!?W?2?1@R      ??!       Z	?W?2?1@?W?2?1@!?W?2?1@JCPU_ONLYY?$????Q@b 