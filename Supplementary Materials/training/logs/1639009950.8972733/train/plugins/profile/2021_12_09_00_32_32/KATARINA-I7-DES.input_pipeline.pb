	??+e?????+e???!??+e???	DؽIL,??DؽIL,??!DؽIL,??"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$??+e???<Nё\???A"?uq??Y?&S???*	     L?@2g
0Iterator::Model::Prefetch::FlatMap[0]::Generator??_?L??!|??G?jX@)??_?L??1|??G?jX@:Preprocessing2F
Iterator::Model?&S???!k8%?'? @)S?!?uq??1??R΀???:Preprocessing2P
Iterator::Model::Prefetcha2U0*?s?!q ?;????)a2U0*?s?1q ?;????:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMap??^??!=?N¦zX@)/n??b?1g?E?z??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 4.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9DؽIL,??>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	<Nё\???<Nё\???!<Nё\???      ??!       "      ??!       *      ??!       2	"?uq??"?uq??!"?uq??:      ??!       B      ??!       J	?&S????&S???!?&S???R      ??!       Z	?&S????&S???!?&S???JCPU_ONLYYDؽIL,??b 