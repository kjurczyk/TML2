	??6 -@??6 -@!??6 -@	XEV??D@XEV??D@!XEV??D@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$??6 -@Έ?????A?St$?? @YԚ??(@*	????L??@2P
Iterator::Model::Prefetch?{??P?@!X?j???G@)?{??P?@1X?j???G@:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMaps??A@!?1??J@)6<?R?!@11?_z@@:Preprocessing2g
0Iterator::Model::Prefetch::FlatMap[0]::Generator c?ZB>@!?Z?X=4@) c?ZB>@1?Z?X=4@:Preprocessing2F
Iterator::Model&S?@!c??q.?G@)1?*?Թ?1?],???:Preprocessing2e
.Iterator::Model::Prefetch::FlatMap::FromTensora2U0*?S?!t?3????)a2U0*?S?1t?3????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 41.5% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*no9YEV??D@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	Έ?????Έ?????!Έ?????      ??!       "      ??!       *      ??!       2	?St$?? @?St$?? @!?St$?? @:      ??!       B      ??!       J	Ԛ??(@Ԛ??(@!Ԛ??(@R      ??!       Z	Ԛ??(@Ԛ??(@!Ԛ??(@JCPU_ONLYYYEV??D@b 