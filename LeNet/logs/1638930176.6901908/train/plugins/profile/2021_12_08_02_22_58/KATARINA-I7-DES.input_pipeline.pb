	??h o?@??h o?@!??h o?@	?E?ǉ???E?ǉ??!?E?ǉ??"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$??h o?@??e??a??A?e??a?@YHP?sע?*	33333?@2g
0Iterator::Model::Prefetch::FlatMap[0]::Generator?HP???!????kX@)?HP???1????kX@:Preprocessing2F
Iterator::Modeln????!????
? @)??Pk?w??1?9GJ???:Preprocessing2P
Iterator::Model::Prefetch?+e?Xw?!7wE????)?+e?Xw?17wE????:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMap??^)??!*X©?yX@)????Mb`?17ݤ????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 3.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?E?ǉ??>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??e??a????e??a??!??e??a??      ??!       "      ??!       *      ??!       2	?e??a?@?e??a?@!?e??a?@:      ??!       B      ??!       J	HP?sע?HP?sע?!HP?sע?R      ??!       Z	HP?sע?HP?sע?!HP?sע?JCPU_ONLYY?E?ǉ??b 