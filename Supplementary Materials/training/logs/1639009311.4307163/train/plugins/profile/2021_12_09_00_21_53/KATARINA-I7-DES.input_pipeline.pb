	??	h"l@??	h"l@!??	h"l@	?????k @?????k @!?????k @"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$??	h"l@?-?????A?z?G?@Y???<,Ԫ?*	fffff?}@2g
0Iterator::Model::Prefetch::FlatMap[0]::Generator?c]?F??!%S1xW@)?c]?F??1%S1xW@:Preprocessing2F
Iterator::Model?5?;Nѡ?!?G6??G@)?v??/??1?????@:Preprocessing2P
Iterator::Model::Prefetch ?o_?y?!?z?4??) ?o_?y?1?z?4??:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMap?W?2??!??|??+W@)?????g?1?/$09???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 5.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?????k @>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?-??????-?????!?-?????      ??!       "      ??!       *      ??!       2	?z?G?@?z?G?@!?z?G?@:      ??!       B      ??!       J	???<,Ԫ????<,Ԫ?!???<,Ԫ?R      ??!       Z	???<,Ԫ????<,Ԫ?!???<,Ԫ?JCPU_ONLYY?????k @b 