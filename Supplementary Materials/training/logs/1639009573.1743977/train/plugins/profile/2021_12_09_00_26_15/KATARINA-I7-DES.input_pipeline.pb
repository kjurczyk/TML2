	?c?Z?@?c?Z?@!?c?Z?@	?t???@?t???@!?t???@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?c?Z?@?1w-!??AF????x@Y??u????*	????̴?@2g
0Iterator::Model::Prefetch::FlatMap[0]::Generator?MbX9??!???*B?W@)?MbX9??1???*B?W@:Preprocessing2F
Iterator::Model?W[?????!?!???@)J+???1??rq??
@:Preprocessing2P
Iterator::Model::Prefetch?+e?Xw?!vVE?l??)?+e?Xw?1vVE?l??:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMap????H??!??#s??W@)ŏ1w-!_?1?9.pH???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 3.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?t???@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?1w-!???1w-!??!?1w-!??      ??!       "      ??!       *      ??!       2	F????x@F????x@!F????x@:      ??!       B      ??!       J	??u??????u????!??u????R      ??!       Z	??u??????u????!??u????JCPU_ONLYY?t???@b 