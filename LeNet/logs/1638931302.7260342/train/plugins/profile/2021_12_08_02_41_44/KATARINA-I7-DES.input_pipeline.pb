	ŏ1w-!@ŏ1w-!@!ŏ1w-!@	_?????_?????!_?????"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$ŏ1w-!@??Q????A??x?&1@Y?ڊ?e???*	33333??@2g
0Iterator::Model::Prefetch::FlatMap[0]::Generator??y???!?l$??DW@)??y???1?l$??DW@:Preprocessing2F
Iterator::Model?	h"lx??!?f????@)???3???1q?(˼0@:Preprocessing2P
Iterator::Model::Prefetch?+e?X??!?|?:???)?+e?X??1?|?:???:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMapۊ?e????!?ya6?aW@)??H?}m?1^?=????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 1.5% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9^?????#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??Q??????Q????!??Q????      ??!       "      ??!       *      ??!       2	??x?&1@??x?&1@!??x?&1@:      ??!       B      ??!       J	?ڊ?e????ڊ?e???!?ڊ?e???R      ??!       Z	?ڊ?e????ڊ?e???!?ڊ?e???JCPU_ONLYY^?????b 