	?w??#Y#@?w??#Y#@!?w??#Y#@	O?Z#??F@O?Z#??F@!O?Z#??F@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?w??#Y#@???????A?[ A??@Yn4??@?@*	4333?;?@2P
Iterator::Model::Prefetch`vO*@!???c??F@)`vO*@1???c??F@:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMap??T???@!΄?
?J@) ?~?:?@1C46?@@:Preprocessing2g
0Iterator::Model::Prefetch::FlatMap[0]::Generator?G?z???!????55@)?G?z???1????55@:Preprocessing2F
Iterator::Model?W?r@!1{:??\G@)?V-??1???OdV??:Preprocessing2e
.Iterator::Model::Prefetch::FlatMap::FromTensor????MbP?!tDC? ???)????MbP?1tDC? ???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 45.2% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*no9O?Z#??F@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??????????????!???????      ??!       "      ??!       *      ??!       2	?[ A??@?[ A??@!?[ A??@:      ??!       B      ??!       J	n4??@?@n4??@?@!n4??@?@R      ??!       Z	n4??@?@n4??@?@!n4??@?@JCPU_ONLYYO?Z#??F@b 