	 A?c?]@ A?c?]@! A?c?]@	r?5????r?5????!r?5????"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$ A?c?]@??/?$??A?,C?b@Y;?O??n??*	???????@2g
0Iterator::Model::Prefetch::FlatMap[0]::Generator?G?z???!?Bh.WX@)?G?z???1?Bh.WX@:Preprocessing2F
Iterator::Model?Q?????!?20+s@)??@??ǈ?1?????? @:Preprocessing2P
Iterator::Model::Prefetch??_vOv?!ǰi>?-??)??_vOv?1ǰi>?-??:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMap?N@a???!jo~?f<X@)??_?Le?1?,x??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 3.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9r?5????>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??/?$????/?$??!??/?$??      ??!       "      ??!       *      ??!       2	?,C?b@?,C?b@!?,C?b@:      ??!       B      ??!       J	;?O??n??;?O??n??!;?O??n??R      ??!       Z	;?O??n??;?O??n??!;?O??n??JCPU_ONLYYr?5????b 