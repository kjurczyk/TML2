	'?W?@'?W?@!'?W?@	??^r6?????^r6???!??^r6???"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$'?W?@aTR'????A?e?c]?@Y?{??Pk??*	33333?@2g
0Iterator::Model::Prefetch::FlatMap[0]::Generator?"??~j??!W1??W@)?"??~j??1W1??W@:Preprocessing2F
Iterator::Model?s????!.qԨ?@)j?t???1??s??[@:Preprocessing2P
Iterator::Model::Prefetch??6???!??????)??6???1??????:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMap(??y??!??rU?W@)???_vOn?1???X???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 1.6% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9??^r6???#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	aTR'????aTR'????!aTR'????      ??!       "      ??!       *      ??!       2	?e?c]?@?e?c]?@!?e?c]?@:      ??!       B      ??!       J	?{??Pk???{??Pk??!?{??Pk??R      ??!       Z	?{??Pk???{??Pk??!?{??Pk??JCPU_ONLYY??^r6???b 