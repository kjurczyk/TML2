	?):?????):????!?):????	|c?yv:	@|c?yv:	@!|c?yv:	@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?):???????h o??A??g??s??Y??6???*	?????q?@2g
0Iterator::Model::Prefetch::FlatMap[0]::Generator<Nё\???!?B?a?W@)<Nё\???1?B?a?W@:Preprocessing2F
Iterator::Model}гY????!??Z??@)2U0*???1=	&?2@:Preprocessing2P
Iterator::Model::Prefetch??_?L??!~???o??)??_?L??1~???o??:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMapa??+e??!x?*M? X@)F%u?k?1??ttP$??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 7.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9|c?yv:	@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???h o?????h o??!???h o??      ??!       "      ??!       *      ??!       2	??g??s????g??s??!??g??s??:      ??!       B      ??!       J	??6?????6???!??6???R      ??!       Z	??6?????6???!??6???JCPU_ONLYY|c?yv:	@b 