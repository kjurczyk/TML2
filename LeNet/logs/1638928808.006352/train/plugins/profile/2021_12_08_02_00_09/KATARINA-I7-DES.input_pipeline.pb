	s??A?D@s??A?D@!s??A?D@	*?g????*?g????!*?g????"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$s??A?D@46</@A?0?*:@Y^K?=???*	?????أ@2g
0Iterator::Model::Prefetch::FlatMap[0]::Generator?G?z@!݄????X@)?G?z@1݄????X@:Preprocessing2F
Iterator::Model????????!???7
~??)??d?`T??1??%?k???:Preprocessing2P
Iterator::Model::Prefetch?ZӼ?}?!???=???)?ZӼ?}?1???=???:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMap?=?U@!????X@)?g??s?u?1D????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 37.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9)?g????>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	46</@46</@!46</@      ??!       "      ??!       *      ??!       2	?0?*:@?0?*:@!?0?*:@:      ??!       B      ??!       J	^K?=???^K?=???!^K?=???R      ??!       Z	^K?=???^K?=???!^K?=???JCPU_ONLYY)?g????b 