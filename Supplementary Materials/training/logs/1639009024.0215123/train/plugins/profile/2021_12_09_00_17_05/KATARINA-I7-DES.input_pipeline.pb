	?,C??@?,C??@!?,C??@	??2?̎????2?̎??!??2?̎??"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?,C??@]?Fx??Az6?>W@Y46<???*	???????@2g
0Iterator::Model::Prefetch::FlatMap[0]::Generator???9#J??!5??&W@)???9#J??15??&W@:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMap??m4????!bV@??tX@)?lV}???1??|@??@:Preprocessing2F
Iterator::ModelΈ?????!?3?? o@)????????1
@??n??:Preprocessing2P
Iterator::Model::Prefetch?HP?x?!?N?A????)?HP?x?1?N?A????:Preprocessing2e
.Iterator::Model::Prefetch::FlatMap::FromTensora2U0*?3?!?$????)a2U0*?3?1?$????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 6.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9??2?̎??>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	]?Fx??]?Fx??!]?Fx??      ??!       "      ??!       *      ??!       2	z6?>W@z6?>W@!z6?>W@:      ??!       B      ??!       J	46<???46<???!46<???R      ??!       Z	46<???46<???!46<???JCPU_ONLYY??2?̎??b 