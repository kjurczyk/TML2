	z6?>W@z6?>W@!z6?>W@	A?,ʈ??A?,ʈ??!A?,ʈ??"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$z6?>W@*:??H??A??H.?!@Y??H?}??*	??????z@2g
0Iterator::Model::Prefetch::FlatMap[0]::Generator??y?)??!6??P^X@)??y?)??16??P^X@:Preprocessing2F
Iterator::Model?W[?????!#??X*b@)M??St$??1?f?=@:Preprocessing2P
Iterator::Model::Prefetchŏ1w-!o?!$I?$I???)ŏ1w-!o?1$I?$I???:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMap??ʡE??!O?;??X@)_?Q?[?1d\???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 4.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9A?,ʈ??>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	*:??H??*:??H??!*:??H??      ??!       "      ??!       *      ??!       2	??H.?!@??H.?!@!??H.?!@:      ??!       B      ??!       J	??H?}????H?}??!??H?}??R      ??!       Z	??H?}????H?}??!??H?}??JCPU_ONLYYA?,ʈ??b 