	q?-?@q?-?@!q?-?@	???{?????{??!???{??"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$q?-?@?b?=y??A?5^?I?@YǺ????*	33333#?@2g
0Iterator::Model::Prefetch::FlatMap[0]::Generator??s????!?? 4X@)??s????1?? 4X@:Preprocessing2F
Iterator::ModelL7?A`???!?u,1?@)tF??_??1q?x?g @:Preprocessing2P
Iterator::Model::PrefetchHP?s?r?!???2?\??)HP?s?r?1???2?\??:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMap]?Fx??!Q??wJX@)????Mb`?194?????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 3.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9???{??>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?b?=y???b?=y??!?b?=y??      ??!       "      ??!       *      ??!       2	?5^?I?@?5^?I?@!?5^?I?@:      ??!       B      ??!       J	Ǻ????Ǻ????!Ǻ????R      ??!       Z	Ǻ????Ǻ????!Ǻ????JCPU_ONLYY???{??b 