	q?-P@q?-P@!q?-P@	??;?"O@??;?"O@!??;?"O@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$q?-P@?V?/?'??A?uq`@Y?^)?G@*	ffff?¼@2P
Iterator::Model::Prefetch???@!۽???I@)???@1۽???I@:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMap????@!?g??sgH@)??(\?B@1?_kv9?B@:Preprocessing2g
0Iterator::Model::Prefetch::FlatMap[0]::Generator???(???!??,:?&@)???(???1??,:?&@:Preprocessing2F
Iterator::Modele?`TR'@!D?Fp??I@)+??	h??1S??m????:Preprocessing2e
.Iterator::Model::Prefetch::FlatMap::FromTensor????MbP?!4??,?Ћ?)????MbP?14??,?Ћ?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 62.3% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*no9??;?"O@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?V?/?'???V?/?'??!?V?/?'??      ??!       "      ??!       *      ??!       2	?uq`@?uq`@!?uq`@:      ??!       B      ??!       J	?^)?G@?^)?G@!?^)?G@R      ??!       Z	?^)?G@?^)?G@!?^)?G@JCPU_ONLYY??;?"O@b 