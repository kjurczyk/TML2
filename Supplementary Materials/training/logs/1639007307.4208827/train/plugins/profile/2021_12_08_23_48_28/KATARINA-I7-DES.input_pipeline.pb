	yX?5??%@yX?5??%@!yX?5??%@	?P}??B@?P}??B@!?P}??B@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$yX?5??%@ˡE?????Aףp=
@Y??_??@*	    @??@2P
Iterator::Model::Prefetch???<,@!j?ve?>G@)???<,@1j?ve?>G@:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMap?ݓ??@!???+J@)?????@1ZR?wuE@@:Preprocessing2g
0Iterator::Model::Prefetch::FlatMap[0]::Generator??N@a??!???"i?3@)??N@a??1???"i?3@:Preprocessing2F
Iterator::Model$???~{@!?mb? ?G@)1?*?Թ?1xRq?ϫ??:Preprocessing2e
.Iterator::Model::Prefetch::FlatMap::FromTensor??H?}M?!???O:Q??)??H?}M?1???O:Q??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 37.6% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*no9?P}??B@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	ˡE?????ˡE?????!ˡE?????      ??!       "      ??!       *      ??!       2	ףp=
@ףp=
@!ףp=
@:      ??!       B      ??!       J	??_??@??_??@!??_??@R      ??!       Z	??_??@??_??@!??_??@JCPU_ONLYY?P}??B@b 