	?_vO?@?_vO?@!?_vO?@	ml?J??ml?J??!ml?J??"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?_vO?@2w-!???AΪ??Vl@YaTR'????*	33333{|@2g
0Iterator::Model::Prefetch::FlatMap[0]::Generator+?????!?}T?B?W@)+?????1?}T?B?W@:Preprocessing2F
Iterator::Model;?O??n??!k??W??@)?{??Pk??1??#??@:Preprocessing2P
Iterator::Model::Prefetch??ZӼ?t?!?{?????)??ZӼ?t?1?{?????:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMap?c]?F??!<?B?3X@)-C??6j?1?O5w?x??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 4.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9nl?J??>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	2w-!???2w-!???!2w-!???      ??!       "      ??!       *      ??!       2	Ϊ??Vl@Ϊ??Vl@!Ϊ??Vl@:      ??!       B      ??!       J	aTR'????aTR'????!aTR'????R      ??!       Z	aTR'????aTR'????!aTR'????JCPU_ONLYYnl?J??b 