	???ׁs@???ׁs@!???ׁs@	CK=J @CK=J @!CK=J @"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$???ׁs@?p=
ף??A?8EGr?@Y?#??????*	gffff??@2g
0Iterator::Model::Prefetch::FlatMap[0]::Generator46<?R??!?y:!X@)46<?R??1?y:!X@:Preprocessing2F
Iterator::Model{?G?z??!.??-Y?
@)??H?}??1_?r?+^@:Preprocessing2P
Iterator::Model::PrefetchǺ???v?!?[?̵ ??)Ǻ???v?1?[?̵ ??:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMap??7??d??!?6?(X@)/n??b?1#??????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 4.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9CK=J @>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?p=
ף???p=
ף??!?p=
ף??      ??!       "      ??!       *      ??!       2	?8EGr?@?8EGr?@!?8EGr?@:      ??!       B      ??!       J	?#???????#??????!?#??????R      ??!       Z	?#???????#??????!?#??????JCPU_ONLYYCK=J @b 