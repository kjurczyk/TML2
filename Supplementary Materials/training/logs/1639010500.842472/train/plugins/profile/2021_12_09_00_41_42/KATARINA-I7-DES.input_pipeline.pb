	???&S@???&S@!???&S@	/:?Z"k??/:?Z"k??!/:?Z"k??"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$???&S@-C??6??A*:??H@Y??Pk?w??*	??????|@2g
0Iterator::Model::Prefetch::FlatMap[0]::Generatorq???h ??!?
ZW?W@)q???h ??1?
ZW?W@:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMap???~?:??!?'??&X@)a2U0*???1??1???@:Preprocessing2F
Iterator::ModelX9??v???!Y[|A(@)??_vO??1??'?@:Preprocessing2P
Iterator::Model::PrefetchU???N@s?!NF?4x??)U???N@s?1NF?4x??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 4.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9.:?Z"k??>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	-C??6??-C??6??!-C??6??      ??!       "      ??!       *      ??!       2	*:??H@*:??H@!*:??H@:      ??!       B      ??!       J	??Pk?w????Pk?w??!??Pk?w??R      ??!       Z	??Pk?w????Pk?w??!??Pk?w??JCPU_ONLYY.:?Z"k??b 