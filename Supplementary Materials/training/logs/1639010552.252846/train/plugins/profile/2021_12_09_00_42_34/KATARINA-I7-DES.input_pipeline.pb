	5?8EG?@5?8EG?@!5?8EG?@	?<2w???<2w??!?<2w??"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$5?8EG?@?_?L??A}гY??@Y'???????*	????? ?@2g
0Iterator::Model::Prefetch::FlatMap[0]::Generator?.n????!???#?X@)?.n????1???#?X@:Preprocessing2F
Iterator::Model?z6?>??!\?ϲ?	@)???H??1??m0?@:Preprocessing2P
Iterator::Model::Prefetch_?Q?{?!???|????)_?Q?{?1???|????:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMap ?o_???! %?i?1X@)/n??b?1M9?E???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 4.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?<2w??>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?_?L???_?L??!?_?L??      ??!       "      ??!       *      ??!       2	}гY??@}гY??@!}гY??@:      ??!       B      ??!       J	'???????'???????!'???????R      ??!       Z	'???????'???????!'???????JCPU_ONLYY?<2w??b 