Ҥ,
��
B
AssignVariableOp
resource
value"dtype"
dtypetype�
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
�
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

�
Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
�
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring �
@
StaticRegexFullMatch	
input

output
"
patternstring
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.4.12unknown8��#
�
encodeAa/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameencodeAa/kernel
{
#encodeAa/kernel/Read/ReadVariableOpReadVariableOpencodeAa/kernel*&
_output_shapes
:*
dtype0
r
encodeAa/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameencodeAa/bias
k
!encodeAa/bias/Read/ReadVariableOpReadVariableOpencodeAa/bias*
_output_shapes
:*
dtype0
�
encodeAb/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameencodeAb/kernel
{
#encodeAb/kernel/Read/ReadVariableOpReadVariableOpencodeAb/kernel*&
_output_shapes
:*
dtype0
r
encodeAb/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameencodeAb/bias
k
!encodeAb/bias/Read/ReadVariableOpReadVariableOpencodeAb/bias*
_output_shapes
:*
dtype0
�
encodeBa/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameencodeBa/kernel
{
#encodeBa/kernel/Read/ReadVariableOpReadVariableOpencodeBa/kernel*&
_output_shapes
: *
dtype0
r
encodeBa/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameencodeBa/bias
k
!encodeBa/bias/Read/ReadVariableOpReadVariableOpencodeBa/bias*
_output_shapes
: *
dtype0
�
encodeBb/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  * 
shared_nameencodeBb/kernel
{
#encodeBb/kernel/Read/ReadVariableOpReadVariableOpencodeBb/kernel*&
_output_shapes
:  *
dtype0
r
encodeBb/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameencodeBb/bias
k
!encodeBb/bias/Read/ReadVariableOpReadVariableOpencodeBb/bias*
_output_shapes
: *
dtype0
�
encodeCa/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @* 
shared_nameencodeCa/kernel
{
#encodeCa/kernel/Read/ReadVariableOpReadVariableOpencodeCa/kernel*&
_output_shapes
: @*
dtype0
r
encodeCa/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameencodeCa/bias
k
!encodeCa/bias/Read/ReadVariableOpReadVariableOpencodeCa/bias*
_output_shapes
:@*
dtype0
�
encodeCb/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@* 
shared_nameencodeCb/kernel
{
#encodeCb/kernel/Read/ReadVariableOpReadVariableOpencodeCb/kernel*&
_output_shapes
:@@*
dtype0
r
encodeCb/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameencodeCb/bias
k
!encodeCb/bias/Read/ReadVariableOpReadVariableOpencodeCb/bias*
_output_shapes
:@*
dtype0
�
encodeDa/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@�* 
shared_nameencodeDa/kernel
|
#encodeDa/kernel/Read/ReadVariableOpReadVariableOpencodeDa/kernel*'
_output_shapes
:@�*
dtype0
s
encodeDa/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameencodeDa/bias
l
!encodeDa/bias/Read/ReadVariableOpReadVariableOpencodeDa/bias*
_output_shapes	
:�*
dtype0
�
encodeDb/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��* 
shared_nameencodeDb/kernel
}
#encodeDb/kernel/Read/ReadVariableOpReadVariableOpencodeDb/kernel*(
_output_shapes
:��*
dtype0
s
encodeDb/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameencodeDb/bias
l
!encodeDb/bias/Read/ReadVariableOpReadVariableOpencodeDb/bias*
_output_shapes	
:�*
dtype0
�
encodeEa/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��* 
shared_nameencodeEa/kernel
}
#encodeEa/kernel/Read/ReadVariableOpReadVariableOpencodeEa/kernel*(
_output_shapes
:��*
dtype0
s
encodeEa/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameencodeEa/bias
l
!encodeEa/bias/Read/ReadVariableOpReadVariableOpencodeEa/bias*
_output_shapes	
:�*
dtype0
�
encodeEb/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��* 
shared_nameencodeEb/kernel
}
#encodeEb/kernel/Read/ReadVariableOpReadVariableOpencodeEb/kernel*(
_output_shapes
:��*
dtype0
s
encodeEb/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameencodeEb/bias
l
!encodeEb/bias/Read/ReadVariableOpReadVariableOpencodeEb/bias*
_output_shapes	
:�*
dtype0
�
transconvE/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*"
shared_nametransconvE/kernel
�
%transconvE/kernel/Read/ReadVariableOpReadVariableOptransconvE/kernel*(
_output_shapes
:��*
dtype0
w
transconvE/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�* 
shared_nametransconvE/bias
p
#transconvE/bias/Read/ReadVariableOpReadVariableOptransconvE/bias*
_output_shapes	
:�*
dtype0
�
decodeCa/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��* 
shared_namedecodeCa/kernel
}
#decodeCa/kernel/Read/ReadVariableOpReadVariableOpdecodeCa/kernel*(
_output_shapes
:��*
dtype0
s
decodeCa/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedecodeCa/bias
l
!decodeCa/bias/Read/ReadVariableOpReadVariableOpdecodeCa/bias*
_output_shapes	
:�*
dtype0
�
decodeCb/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��* 
shared_namedecodeCb/kernel
}
#decodeCb/kernel/Read/ReadVariableOpReadVariableOpdecodeCb/kernel*(
_output_shapes
:��*
dtype0
s
decodeCb/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedecodeCb/bias
l
!decodeCb/bias/Read/ReadVariableOpReadVariableOpdecodeCb/bias*
_output_shapes	
:�*
dtype0
�
transconvC/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@�*"
shared_nametransconvC/kernel
�
%transconvC/kernel/Read/ReadVariableOpReadVariableOptransconvC/kernel*'
_output_shapes
:@�*
dtype0
v
transconvC/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@* 
shared_nametransconvC/bias
o
#transconvC/bias/Read/ReadVariableOpReadVariableOptransconvC/bias*
_output_shapes
:@*
dtype0
�
decodeBa/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:�@* 
shared_namedecodeBa/kernel
|
#decodeBa/kernel/Read/ReadVariableOpReadVariableOpdecodeBa/kernel*'
_output_shapes
:�@*
dtype0
r
decodeBa/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedecodeBa/bias
k
!decodeBa/bias/Read/ReadVariableOpReadVariableOpdecodeBa/bias*
_output_shapes
:@*
dtype0
�
decodeBb/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@* 
shared_namedecodeBb/kernel
{
#decodeBb/kernel/Read/ReadVariableOpReadVariableOpdecodeBb/kernel*&
_output_shapes
:@@*
dtype0
r
decodeBb/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedecodeBb/bias
k
!decodeBb/bias/Read/ReadVariableOpReadVariableOpdecodeBb/bias*
_output_shapes
:@*
dtype0
�
transconvB/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*"
shared_nametransconvB/kernel

%transconvB/kernel/Read/ReadVariableOpReadVariableOptransconvB/kernel*&
_output_shapes
: @*
dtype0
v
transconvB/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nametransconvB/bias
o
#transconvB/bias/Read/ReadVariableOpReadVariableOptransconvB/bias*
_output_shapes
: *
dtype0
�
decodeAa/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ * 
shared_namedecodeAa/kernel
{
#decodeAa/kernel/Read/ReadVariableOpReadVariableOpdecodeAa/kernel*&
_output_shapes
:@ *
dtype0
r
decodeAa/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedecodeAa/bias
k
!decodeAa/bias/Read/ReadVariableOpReadVariableOpdecodeAa/bias*
_output_shapes
: *
dtype0
�
decodeAb/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  * 
shared_namedecodeAb/kernel
{
#decodeAb/kernel/Read/ReadVariableOpReadVariableOpdecodeAb/kernel*&
_output_shapes
:  *
dtype0
r
decodeAb/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedecodeAb/bias
k
!decodeAb/bias/Read/ReadVariableOpReadVariableOpdecodeAb/bias*
_output_shapes
: *
dtype0
�
transconvA/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nametransconvA/kernel

%transconvA/kernel/Read/ReadVariableOpReadVariableOptransconvA/kernel*&
_output_shapes
: *
dtype0
v
transconvA/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nametransconvA/bias
o
#transconvA/bias/Read/ReadVariableOpReadVariableOptransconvA/bias*
_output_shapes
:*
dtype0
�
convOuta/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconvOuta/kernel
{
#convOuta/kernel/Read/ReadVariableOpReadVariableOpconvOuta/kernel*&
_output_shapes
: *
dtype0
r
convOuta/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconvOuta/bias
k
!convOuta/bias/Read/ReadVariableOpReadVariableOpconvOuta/bias*
_output_shapes
:*
dtype0
�
convOutb/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconvOutb/kernel
{
#convOutb/kernel/Read/ReadVariableOpReadVariableOpconvOutb/kernel*&
_output_shapes
:*
dtype0
r
convOutb/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconvOutb/bias
k
!convOutb/bias/Read/ReadVariableOpReadVariableOpconvOutb/bias*
_output_shapes
:*
dtype0
�
PredictionMask/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_namePredictionMask/kernel
�
)PredictionMask/kernel/Read/ReadVariableOpReadVariableOpPredictionMask/kernel*&
_output_shapes
:*
dtype0
~
PredictionMask/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_namePredictionMask/bias
w
'PredictionMask/bias/Read/ReadVariableOpReadVariableOpPredictionMask/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_2
[
total_2/Read/ReadVariableOpReadVariableOptotal_2*
_output_shapes
: *
dtype0
b
count_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
dtype0
�
Adam/encodeAa/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/encodeAa/kernel/m
�
*Adam/encodeAa/kernel/m/Read/ReadVariableOpReadVariableOpAdam/encodeAa/kernel/m*&
_output_shapes
:*
dtype0
�
Adam/encodeAa/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/encodeAa/bias/m
y
(Adam/encodeAa/bias/m/Read/ReadVariableOpReadVariableOpAdam/encodeAa/bias/m*
_output_shapes
:*
dtype0
�
Adam/encodeAb/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/encodeAb/kernel/m
�
*Adam/encodeAb/kernel/m/Read/ReadVariableOpReadVariableOpAdam/encodeAb/kernel/m*&
_output_shapes
:*
dtype0
�
Adam/encodeAb/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/encodeAb/bias/m
y
(Adam/encodeAb/bias/m/Read/ReadVariableOpReadVariableOpAdam/encodeAb/bias/m*
_output_shapes
:*
dtype0
�
Adam/encodeBa/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/encodeBa/kernel/m
�
*Adam/encodeBa/kernel/m/Read/ReadVariableOpReadVariableOpAdam/encodeBa/kernel/m*&
_output_shapes
: *
dtype0
�
Adam/encodeBa/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/encodeBa/bias/m
y
(Adam/encodeBa/bias/m/Read/ReadVariableOpReadVariableOpAdam/encodeBa/bias/m*
_output_shapes
: *
dtype0
�
Adam/encodeBb/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *'
shared_nameAdam/encodeBb/kernel/m
�
*Adam/encodeBb/kernel/m/Read/ReadVariableOpReadVariableOpAdam/encodeBb/kernel/m*&
_output_shapes
:  *
dtype0
�
Adam/encodeBb/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/encodeBb/bias/m
y
(Adam/encodeBb/bias/m/Read/ReadVariableOpReadVariableOpAdam/encodeBb/bias/m*
_output_shapes
: *
dtype0
�
Adam/encodeCa/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*'
shared_nameAdam/encodeCa/kernel/m
�
*Adam/encodeCa/kernel/m/Read/ReadVariableOpReadVariableOpAdam/encodeCa/kernel/m*&
_output_shapes
: @*
dtype0
�
Adam/encodeCa/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/encodeCa/bias/m
y
(Adam/encodeCa/bias/m/Read/ReadVariableOpReadVariableOpAdam/encodeCa/bias/m*
_output_shapes
:@*
dtype0
�
Adam/encodeCb/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*'
shared_nameAdam/encodeCb/kernel/m
�
*Adam/encodeCb/kernel/m/Read/ReadVariableOpReadVariableOpAdam/encodeCb/kernel/m*&
_output_shapes
:@@*
dtype0
�
Adam/encodeCb/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/encodeCb/bias/m
y
(Adam/encodeCb/bias/m/Read/ReadVariableOpReadVariableOpAdam/encodeCb/bias/m*
_output_shapes
:@*
dtype0
�
Adam/encodeDa/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@�*'
shared_nameAdam/encodeDa/kernel/m
�
*Adam/encodeDa/kernel/m/Read/ReadVariableOpReadVariableOpAdam/encodeDa/kernel/m*'
_output_shapes
:@�*
dtype0
�
Adam/encodeDa/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/encodeDa/bias/m
z
(Adam/encodeDa/bias/m/Read/ReadVariableOpReadVariableOpAdam/encodeDa/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/encodeDb/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*'
shared_nameAdam/encodeDb/kernel/m
�
*Adam/encodeDb/kernel/m/Read/ReadVariableOpReadVariableOpAdam/encodeDb/kernel/m*(
_output_shapes
:��*
dtype0
�
Adam/encodeDb/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/encodeDb/bias/m
z
(Adam/encodeDb/bias/m/Read/ReadVariableOpReadVariableOpAdam/encodeDb/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/encodeEa/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*'
shared_nameAdam/encodeEa/kernel/m
�
*Adam/encodeEa/kernel/m/Read/ReadVariableOpReadVariableOpAdam/encodeEa/kernel/m*(
_output_shapes
:��*
dtype0
�
Adam/encodeEa/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/encodeEa/bias/m
z
(Adam/encodeEa/bias/m/Read/ReadVariableOpReadVariableOpAdam/encodeEa/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/encodeEb/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*'
shared_nameAdam/encodeEb/kernel/m
�
*Adam/encodeEb/kernel/m/Read/ReadVariableOpReadVariableOpAdam/encodeEb/kernel/m*(
_output_shapes
:��*
dtype0
�
Adam/encodeEb/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/encodeEb/bias/m
z
(Adam/encodeEb/bias/m/Read/ReadVariableOpReadVariableOpAdam/encodeEb/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/transconvE/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*)
shared_nameAdam/transconvE/kernel/m
�
,Adam/transconvE/kernel/m/Read/ReadVariableOpReadVariableOpAdam/transconvE/kernel/m*(
_output_shapes
:��*
dtype0
�
Adam/transconvE/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*'
shared_nameAdam/transconvE/bias/m
~
*Adam/transconvE/bias/m/Read/ReadVariableOpReadVariableOpAdam/transconvE/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/decodeCa/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*'
shared_nameAdam/decodeCa/kernel/m
�
*Adam/decodeCa/kernel/m/Read/ReadVariableOpReadVariableOpAdam/decodeCa/kernel/m*(
_output_shapes
:��*
dtype0
�
Adam/decodeCa/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/decodeCa/bias/m
z
(Adam/decodeCa/bias/m/Read/ReadVariableOpReadVariableOpAdam/decodeCa/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/decodeCb/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*'
shared_nameAdam/decodeCb/kernel/m
�
*Adam/decodeCb/kernel/m/Read/ReadVariableOpReadVariableOpAdam/decodeCb/kernel/m*(
_output_shapes
:��*
dtype0
�
Adam/decodeCb/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/decodeCb/bias/m
z
(Adam/decodeCb/bias/m/Read/ReadVariableOpReadVariableOpAdam/decodeCb/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/transconvC/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@�*)
shared_nameAdam/transconvC/kernel/m
�
,Adam/transconvC/kernel/m/Read/ReadVariableOpReadVariableOpAdam/transconvC/kernel/m*'
_output_shapes
:@�*
dtype0
�
Adam/transconvC/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_nameAdam/transconvC/bias/m
}
*Adam/transconvC/bias/m/Read/ReadVariableOpReadVariableOpAdam/transconvC/bias/m*
_output_shapes
:@*
dtype0
�
Adam/decodeBa/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�@*'
shared_nameAdam/decodeBa/kernel/m
�
*Adam/decodeBa/kernel/m/Read/ReadVariableOpReadVariableOpAdam/decodeBa/kernel/m*'
_output_shapes
:�@*
dtype0
�
Adam/decodeBa/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/decodeBa/bias/m
y
(Adam/decodeBa/bias/m/Read/ReadVariableOpReadVariableOpAdam/decodeBa/bias/m*
_output_shapes
:@*
dtype0
�
Adam/decodeBb/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*'
shared_nameAdam/decodeBb/kernel/m
�
*Adam/decodeBb/kernel/m/Read/ReadVariableOpReadVariableOpAdam/decodeBb/kernel/m*&
_output_shapes
:@@*
dtype0
�
Adam/decodeBb/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/decodeBb/bias/m
y
(Adam/decodeBb/bias/m/Read/ReadVariableOpReadVariableOpAdam/decodeBb/bias/m*
_output_shapes
:@*
dtype0
�
Adam/transconvB/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*)
shared_nameAdam/transconvB/kernel/m
�
,Adam/transconvB/kernel/m/Read/ReadVariableOpReadVariableOpAdam/transconvB/kernel/m*&
_output_shapes
: @*
dtype0
�
Adam/transconvB/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/transconvB/bias/m
}
*Adam/transconvB/bias/m/Read/ReadVariableOpReadVariableOpAdam/transconvB/bias/m*
_output_shapes
: *
dtype0
�
Adam/decodeAa/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *'
shared_nameAdam/decodeAa/kernel/m
�
*Adam/decodeAa/kernel/m/Read/ReadVariableOpReadVariableOpAdam/decodeAa/kernel/m*&
_output_shapes
:@ *
dtype0
�
Adam/decodeAa/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/decodeAa/bias/m
y
(Adam/decodeAa/bias/m/Read/ReadVariableOpReadVariableOpAdam/decodeAa/bias/m*
_output_shapes
: *
dtype0
�
Adam/decodeAb/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *'
shared_nameAdam/decodeAb/kernel/m
�
*Adam/decodeAb/kernel/m/Read/ReadVariableOpReadVariableOpAdam/decodeAb/kernel/m*&
_output_shapes
:  *
dtype0
�
Adam/decodeAb/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/decodeAb/bias/m
y
(Adam/decodeAb/bias/m/Read/ReadVariableOpReadVariableOpAdam/decodeAb/bias/m*
_output_shapes
: *
dtype0
�
Adam/transconvA/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_nameAdam/transconvA/kernel/m
�
,Adam/transconvA/kernel/m/Read/ReadVariableOpReadVariableOpAdam/transconvA/kernel/m*&
_output_shapes
: *
dtype0
�
Adam/transconvA/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/transconvA/bias/m
}
*Adam/transconvA/bias/m/Read/ReadVariableOpReadVariableOpAdam/transconvA/bias/m*
_output_shapes
:*
dtype0
�
Adam/convOuta/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/convOuta/kernel/m
�
*Adam/convOuta/kernel/m/Read/ReadVariableOpReadVariableOpAdam/convOuta/kernel/m*&
_output_shapes
: *
dtype0
�
Adam/convOuta/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/convOuta/bias/m
y
(Adam/convOuta/bias/m/Read/ReadVariableOpReadVariableOpAdam/convOuta/bias/m*
_output_shapes
:*
dtype0
�
Adam/convOutb/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/convOutb/kernel/m
�
*Adam/convOutb/kernel/m/Read/ReadVariableOpReadVariableOpAdam/convOutb/kernel/m*&
_output_shapes
:*
dtype0
�
Adam/convOutb/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/convOutb/bias/m
y
(Adam/convOutb/bias/m/Read/ReadVariableOpReadVariableOpAdam/convOutb/bias/m*
_output_shapes
:*
dtype0
�
Adam/PredictionMask/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameAdam/PredictionMask/kernel/m
�
0Adam/PredictionMask/kernel/m/Read/ReadVariableOpReadVariableOpAdam/PredictionMask/kernel/m*&
_output_shapes
:*
dtype0
�
Adam/PredictionMask/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameAdam/PredictionMask/bias/m
�
.Adam/PredictionMask/bias/m/Read/ReadVariableOpReadVariableOpAdam/PredictionMask/bias/m*
_output_shapes
:*
dtype0
�
Adam/encodeAa/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/encodeAa/kernel/v
�
*Adam/encodeAa/kernel/v/Read/ReadVariableOpReadVariableOpAdam/encodeAa/kernel/v*&
_output_shapes
:*
dtype0
�
Adam/encodeAa/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/encodeAa/bias/v
y
(Adam/encodeAa/bias/v/Read/ReadVariableOpReadVariableOpAdam/encodeAa/bias/v*
_output_shapes
:*
dtype0
�
Adam/encodeAb/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/encodeAb/kernel/v
�
*Adam/encodeAb/kernel/v/Read/ReadVariableOpReadVariableOpAdam/encodeAb/kernel/v*&
_output_shapes
:*
dtype0
�
Adam/encodeAb/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/encodeAb/bias/v
y
(Adam/encodeAb/bias/v/Read/ReadVariableOpReadVariableOpAdam/encodeAb/bias/v*
_output_shapes
:*
dtype0
�
Adam/encodeBa/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/encodeBa/kernel/v
�
*Adam/encodeBa/kernel/v/Read/ReadVariableOpReadVariableOpAdam/encodeBa/kernel/v*&
_output_shapes
: *
dtype0
�
Adam/encodeBa/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/encodeBa/bias/v
y
(Adam/encodeBa/bias/v/Read/ReadVariableOpReadVariableOpAdam/encodeBa/bias/v*
_output_shapes
: *
dtype0
�
Adam/encodeBb/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *'
shared_nameAdam/encodeBb/kernel/v
�
*Adam/encodeBb/kernel/v/Read/ReadVariableOpReadVariableOpAdam/encodeBb/kernel/v*&
_output_shapes
:  *
dtype0
�
Adam/encodeBb/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/encodeBb/bias/v
y
(Adam/encodeBb/bias/v/Read/ReadVariableOpReadVariableOpAdam/encodeBb/bias/v*
_output_shapes
: *
dtype0
�
Adam/encodeCa/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*'
shared_nameAdam/encodeCa/kernel/v
�
*Adam/encodeCa/kernel/v/Read/ReadVariableOpReadVariableOpAdam/encodeCa/kernel/v*&
_output_shapes
: @*
dtype0
�
Adam/encodeCa/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/encodeCa/bias/v
y
(Adam/encodeCa/bias/v/Read/ReadVariableOpReadVariableOpAdam/encodeCa/bias/v*
_output_shapes
:@*
dtype0
�
Adam/encodeCb/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*'
shared_nameAdam/encodeCb/kernel/v
�
*Adam/encodeCb/kernel/v/Read/ReadVariableOpReadVariableOpAdam/encodeCb/kernel/v*&
_output_shapes
:@@*
dtype0
�
Adam/encodeCb/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/encodeCb/bias/v
y
(Adam/encodeCb/bias/v/Read/ReadVariableOpReadVariableOpAdam/encodeCb/bias/v*
_output_shapes
:@*
dtype0
�
Adam/encodeDa/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@�*'
shared_nameAdam/encodeDa/kernel/v
�
*Adam/encodeDa/kernel/v/Read/ReadVariableOpReadVariableOpAdam/encodeDa/kernel/v*'
_output_shapes
:@�*
dtype0
�
Adam/encodeDa/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/encodeDa/bias/v
z
(Adam/encodeDa/bias/v/Read/ReadVariableOpReadVariableOpAdam/encodeDa/bias/v*
_output_shapes	
:�*
dtype0
�
Adam/encodeDb/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*'
shared_nameAdam/encodeDb/kernel/v
�
*Adam/encodeDb/kernel/v/Read/ReadVariableOpReadVariableOpAdam/encodeDb/kernel/v*(
_output_shapes
:��*
dtype0
�
Adam/encodeDb/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/encodeDb/bias/v
z
(Adam/encodeDb/bias/v/Read/ReadVariableOpReadVariableOpAdam/encodeDb/bias/v*
_output_shapes	
:�*
dtype0
�
Adam/encodeEa/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*'
shared_nameAdam/encodeEa/kernel/v
�
*Adam/encodeEa/kernel/v/Read/ReadVariableOpReadVariableOpAdam/encodeEa/kernel/v*(
_output_shapes
:��*
dtype0
�
Adam/encodeEa/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/encodeEa/bias/v
z
(Adam/encodeEa/bias/v/Read/ReadVariableOpReadVariableOpAdam/encodeEa/bias/v*
_output_shapes	
:�*
dtype0
�
Adam/encodeEb/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*'
shared_nameAdam/encodeEb/kernel/v
�
*Adam/encodeEb/kernel/v/Read/ReadVariableOpReadVariableOpAdam/encodeEb/kernel/v*(
_output_shapes
:��*
dtype0
�
Adam/encodeEb/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/encodeEb/bias/v
z
(Adam/encodeEb/bias/v/Read/ReadVariableOpReadVariableOpAdam/encodeEb/bias/v*
_output_shapes	
:�*
dtype0
�
Adam/transconvE/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*)
shared_nameAdam/transconvE/kernel/v
�
,Adam/transconvE/kernel/v/Read/ReadVariableOpReadVariableOpAdam/transconvE/kernel/v*(
_output_shapes
:��*
dtype0
�
Adam/transconvE/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*'
shared_nameAdam/transconvE/bias/v
~
*Adam/transconvE/bias/v/Read/ReadVariableOpReadVariableOpAdam/transconvE/bias/v*
_output_shapes	
:�*
dtype0
�
Adam/decodeCa/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*'
shared_nameAdam/decodeCa/kernel/v
�
*Adam/decodeCa/kernel/v/Read/ReadVariableOpReadVariableOpAdam/decodeCa/kernel/v*(
_output_shapes
:��*
dtype0
�
Adam/decodeCa/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/decodeCa/bias/v
z
(Adam/decodeCa/bias/v/Read/ReadVariableOpReadVariableOpAdam/decodeCa/bias/v*
_output_shapes	
:�*
dtype0
�
Adam/decodeCb/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*'
shared_nameAdam/decodeCb/kernel/v
�
*Adam/decodeCb/kernel/v/Read/ReadVariableOpReadVariableOpAdam/decodeCb/kernel/v*(
_output_shapes
:��*
dtype0
�
Adam/decodeCb/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/decodeCb/bias/v
z
(Adam/decodeCb/bias/v/Read/ReadVariableOpReadVariableOpAdam/decodeCb/bias/v*
_output_shapes	
:�*
dtype0
�
Adam/transconvC/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@�*)
shared_nameAdam/transconvC/kernel/v
�
,Adam/transconvC/kernel/v/Read/ReadVariableOpReadVariableOpAdam/transconvC/kernel/v*'
_output_shapes
:@�*
dtype0
�
Adam/transconvC/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_nameAdam/transconvC/bias/v
}
*Adam/transconvC/bias/v/Read/ReadVariableOpReadVariableOpAdam/transconvC/bias/v*
_output_shapes
:@*
dtype0
�
Adam/decodeBa/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�@*'
shared_nameAdam/decodeBa/kernel/v
�
*Adam/decodeBa/kernel/v/Read/ReadVariableOpReadVariableOpAdam/decodeBa/kernel/v*'
_output_shapes
:�@*
dtype0
�
Adam/decodeBa/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/decodeBa/bias/v
y
(Adam/decodeBa/bias/v/Read/ReadVariableOpReadVariableOpAdam/decodeBa/bias/v*
_output_shapes
:@*
dtype0
�
Adam/decodeBb/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*'
shared_nameAdam/decodeBb/kernel/v
�
*Adam/decodeBb/kernel/v/Read/ReadVariableOpReadVariableOpAdam/decodeBb/kernel/v*&
_output_shapes
:@@*
dtype0
�
Adam/decodeBb/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/decodeBb/bias/v
y
(Adam/decodeBb/bias/v/Read/ReadVariableOpReadVariableOpAdam/decodeBb/bias/v*
_output_shapes
:@*
dtype0
�
Adam/transconvB/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*)
shared_nameAdam/transconvB/kernel/v
�
,Adam/transconvB/kernel/v/Read/ReadVariableOpReadVariableOpAdam/transconvB/kernel/v*&
_output_shapes
: @*
dtype0
�
Adam/transconvB/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/transconvB/bias/v
}
*Adam/transconvB/bias/v/Read/ReadVariableOpReadVariableOpAdam/transconvB/bias/v*
_output_shapes
: *
dtype0
�
Adam/decodeAa/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *'
shared_nameAdam/decodeAa/kernel/v
�
*Adam/decodeAa/kernel/v/Read/ReadVariableOpReadVariableOpAdam/decodeAa/kernel/v*&
_output_shapes
:@ *
dtype0
�
Adam/decodeAa/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/decodeAa/bias/v
y
(Adam/decodeAa/bias/v/Read/ReadVariableOpReadVariableOpAdam/decodeAa/bias/v*
_output_shapes
: *
dtype0
�
Adam/decodeAb/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *'
shared_nameAdam/decodeAb/kernel/v
�
*Adam/decodeAb/kernel/v/Read/ReadVariableOpReadVariableOpAdam/decodeAb/kernel/v*&
_output_shapes
:  *
dtype0
�
Adam/decodeAb/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/decodeAb/bias/v
y
(Adam/decodeAb/bias/v/Read/ReadVariableOpReadVariableOpAdam/decodeAb/bias/v*
_output_shapes
: *
dtype0
�
Adam/transconvA/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_nameAdam/transconvA/kernel/v
�
,Adam/transconvA/kernel/v/Read/ReadVariableOpReadVariableOpAdam/transconvA/kernel/v*&
_output_shapes
: *
dtype0
�
Adam/transconvA/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/transconvA/bias/v
}
*Adam/transconvA/bias/v/Read/ReadVariableOpReadVariableOpAdam/transconvA/bias/v*
_output_shapes
:*
dtype0
�
Adam/convOuta/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/convOuta/kernel/v
�
*Adam/convOuta/kernel/v/Read/ReadVariableOpReadVariableOpAdam/convOuta/kernel/v*&
_output_shapes
: *
dtype0
�
Adam/convOuta/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/convOuta/bias/v
y
(Adam/convOuta/bias/v/Read/ReadVariableOpReadVariableOpAdam/convOuta/bias/v*
_output_shapes
:*
dtype0
�
Adam/convOutb/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/convOutb/kernel/v
�
*Adam/convOutb/kernel/v/Read/ReadVariableOpReadVariableOpAdam/convOutb/kernel/v*&
_output_shapes
:*
dtype0
�
Adam/convOutb/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/convOutb/bias/v
y
(Adam/convOutb/bias/v/Read/ReadVariableOpReadVariableOpAdam/convOutb/bias/v*
_output_shapes
:*
dtype0
�
Adam/PredictionMask/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameAdam/PredictionMask/kernel/v
�
0Adam/PredictionMask/kernel/v/Read/ReadVariableOpReadVariableOpAdam/PredictionMask/kernel/v*&
_output_shapes
:*
dtype0
�
Adam/PredictionMask/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameAdam/PredictionMask/bias/v
�
.Adam/PredictionMask/bias/v/Read/ReadVariableOpReadVariableOpAdam/PredictionMask/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
��
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*��
value��B�� B��
�	
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
	layer-8

layer_with_weights-5

layer-9
layer-10
layer_with_weights-6
layer-11
layer-12
layer_with_weights-7
layer-13
layer-14
layer_with_weights-8
layer-15
layer_with_weights-9
layer-16
layer_with_weights-10
layer-17
layer-18
layer_with_weights-11
layer-19
layer_with_weights-12
layer-20
layer_with_weights-13
layer-21
layer-22
layer_with_weights-14
layer-23
layer_with_weights-15
layer-24
layer_with_weights-16
layer-25
layer-26
layer_with_weights-17
layer-27
layer_with_weights-18
layer-28
layer_with_weights-19
layer-29
layer-30
 layer_with_weights-20
 layer-31
!layer_with_weights-21
!layer-32
"layer_with_weights-22
"layer-33
#	optimizer
$regularization_losses
%	variables
&trainable_variables
'	keras_api
(
signatures
 
h

)kernel
*bias
+regularization_losses
,	variables
-trainable_variables
.	keras_api
h

/kernel
0bias
1regularization_losses
2	variables
3trainable_variables
4	keras_api
R
5regularization_losses
6	variables
7trainable_variables
8	keras_api
h

9kernel
:bias
;regularization_losses
<	variables
=trainable_variables
>	keras_api
h

?kernel
@bias
Aregularization_losses
B	variables
Ctrainable_variables
D	keras_api
R
Eregularization_losses
F	variables
Gtrainable_variables
H	keras_api
h

Ikernel
Jbias
Kregularization_losses
L	variables
Mtrainable_variables
N	keras_api
R
Oregularization_losses
P	variables
Qtrainable_variables
R	keras_api
h

Skernel
Tbias
Uregularization_losses
V	variables
Wtrainable_variables
X	keras_api
R
Yregularization_losses
Z	variables
[trainable_variables
\	keras_api
h

]kernel
^bias
_regularization_losses
`	variables
atrainable_variables
b	keras_api
R
cregularization_losses
d	variables
etrainable_variables
f	keras_api
h

gkernel
hbias
iregularization_losses
j	variables
ktrainable_variables
l	keras_api
R
mregularization_losses
n	variables
otrainable_variables
p	keras_api
h

qkernel
rbias
sregularization_losses
t	variables
utrainable_variables
v	keras_api
h

wkernel
xbias
yregularization_losses
z	variables
{trainable_variables
|	keras_api
k

}kernel
~bias
regularization_losses
�	variables
�trainable_variables
�	keras_api
V
�regularization_losses
�	variables
�trainable_variables
�	keras_api
n
�kernel
	�bias
�regularization_losses
�	variables
�trainable_variables
�	keras_api
n
�kernel
	�bias
�regularization_losses
�	variables
�trainable_variables
�	keras_api
n
�kernel
	�bias
�regularization_losses
�	variables
�trainable_variables
�	keras_api
V
�regularization_losses
�	variables
�trainable_variables
�	keras_api
n
�kernel
	�bias
�regularization_losses
�	variables
�trainable_variables
�	keras_api
n
�kernel
	�bias
�regularization_losses
�	variables
�trainable_variables
�	keras_api
n
�kernel
	�bias
�regularization_losses
�	variables
�trainable_variables
�	keras_api
V
�regularization_losses
�	variables
�trainable_variables
�	keras_api
n
�kernel
	�bias
�regularization_losses
�	variables
�trainable_variables
�	keras_api
n
�kernel
	�bias
�regularization_losses
�	variables
�trainable_variables
�	keras_api
n
�kernel
	�bias
�regularization_losses
�	variables
�trainable_variables
�	keras_api
V
�regularization_losses
�	variables
�trainable_variables
�	keras_api
n
�kernel
	�bias
�regularization_losses
�	variables
�trainable_variables
�	keras_api
n
�kernel
	�bias
�regularization_losses
�	variables
�trainable_variables
�	keras_api
n
�kernel
	�bias
�regularization_losses
�	variables
�trainable_variables
�	keras_api
�
	�iter
�beta_1
�beta_2

�decay
�learning_rate)m�*m�/m�0m�9m�:m�?m�@m�Im�Jm�Sm�Tm�]m�^m�gm�hm�qm�rm�wm�xm�}m�~m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�)v�*v�/v�0v�9v�:v�?v�@v�Iv�Jv�Sv�Tv�]v�^v�gv�hv�qv�rv�wv�xv�}v�~v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�
 
�
)0
*1
/2
03
94
:5
?6
@7
I8
J9
S10
T11
]12
^13
g14
h15
q16
r17
w18
x19
}20
~21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39
�40
�41
�42
�43
�44
�45
�
)0
*1
/2
03
94
:5
?6
@7
I8
J9
S10
T11
]12
^13
g14
h15
q16
r17
w18
x19
}20
~21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39
�40
�41
�42
�43
�44
�45
�
$regularization_losses
 �layer_regularization_losses
%	variables
�metrics
&trainable_variables
�non_trainable_variables
�layers
�layer_metrics
 
[Y
VARIABLE_VALUEencodeAa/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEencodeAa/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

)0
*1

)0
*1
�
+regularization_losses
 �layer_regularization_losses
,	variables
�metrics
-trainable_variables
�non_trainable_variables
�layers
�layer_metrics
[Y
VARIABLE_VALUEencodeAb/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEencodeAb/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

/0
01

/0
01
�
1regularization_losses
 �layer_regularization_losses
2	variables
�metrics
3trainable_variables
�non_trainable_variables
�layers
�layer_metrics
 
 
 
�
5regularization_losses
 �layer_regularization_losses
6	variables
�metrics
7trainable_variables
�non_trainable_variables
�layers
�layer_metrics
[Y
VARIABLE_VALUEencodeBa/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEencodeBa/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

90
:1

90
:1
�
;regularization_losses
 �layer_regularization_losses
<	variables
�metrics
=trainable_variables
�non_trainable_variables
�layers
�layer_metrics
[Y
VARIABLE_VALUEencodeBb/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEencodeBb/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

?0
@1

?0
@1
�
Aregularization_losses
 �layer_regularization_losses
B	variables
�metrics
Ctrainable_variables
�non_trainable_variables
�layers
�layer_metrics
 
 
 
�
Eregularization_losses
 �layer_regularization_losses
F	variables
�metrics
Gtrainable_variables
�non_trainable_variables
�layers
�layer_metrics
[Y
VARIABLE_VALUEencodeCa/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEencodeCa/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

I0
J1

I0
J1
�
Kregularization_losses
 �layer_regularization_losses
L	variables
�metrics
Mtrainable_variables
�non_trainable_variables
�layers
�layer_metrics
 
 
 
�
Oregularization_losses
 �layer_regularization_losses
P	variables
�metrics
Qtrainable_variables
�non_trainable_variables
�layers
�layer_metrics
[Y
VARIABLE_VALUEencodeCb/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEencodeCb/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE
 

S0
T1

S0
T1
�
Uregularization_losses
 �layer_regularization_losses
V	variables
�metrics
Wtrainable_variables
�non_trainable_variables
�layers
�layer_metrics
 
 
 
�
Yregularization_losses
 �layer_regularization_losses
Z	variables
�metrics
[trainable_variables
�non_trainable_variables
�layers
�layer_metrics
[Y
VARIABLE_VALUEencodeDa/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEencodeDa/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE
 

]0
^1

]0
^1
�
_regularization_losses
 �layer_regularization_losses
`	variables
�metrics
atrainable_variables
�non_trainable_variables
�layers
�layer_metrics
 
 
 
�
cregularization_losses
 �layer_regularization_losses
d	variables
�metrics
etrainable_variables
�non_trainable_variables
�layers
�layer_metrics
[Y
VARIABLE_VALUEencodeDb/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEencodeDb/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE
 

g0
h1

g0
h1
�
iregularization_losses
 �layer_regularization_losses
j	variables
�metrics
ktrainable_variables
�non_trainable_variables
�layers
�layer_metrics
 
 
 
�
mregularization_losses
 �layer_regularization_losses
n	variables
�metrics
otrainable_variables
�non_trainable_variables
�layers
�layer_metrics
[Y
VARIABLE_VALUEencodeEa/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEencodeEa/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE
 

q0
r1

q0
r1
�
sregularization_losses
 �layer_regularization_losses
t	variables
�metrics
utrainable_variables
�non_trainable_variables
�layers
�layer_metrics
[Y
VARIABLE_VALUEencodeEb/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEencodeEb/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE
 

w0
x1

w0
x1
�
yregularization_losses
 �layer_regularization_losses
z	variables
�metrics
{trainable_variables
�non_trainable_variables
�layers
�layer_metrics
^\
VARIABLE_VALUEtransconvE/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEtransconvE/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE
 

}0
~1

}0
~1
�
regularization_losses
 �layer_regularization_losses
�	variables
�metrics
�trainable_variables
�non_trainable_variables
�layers
�layer_metrics
 
 
 
�
�regularization_losses
 �layer_regularization_losses
�	variables
�metrics
�trainable_variables
�non_trainable_variables
�layers
�layer_metrics
\Z
VARIABLE_VALUEdecodeCa/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdecodeCa/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE
 

�0
�1

�0
�1
�
�regularization_losses
 �layer_regularization_losses
�	variables
�metrics
�trainable_variables
�non_trainable_variables
�layers
�layer_metrics
\Z
VARIABLE_VALUEdecodeCb/kernel7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdecodeCb/bias5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUE
 

�0
�1

�0
�1
�
�regularization_losses
 �layer_regularization_losses
�	variables
�metrics
�trainable_variables
�non_trainable_variables
�layers
�layer_metrics
^\
VARIABLE_VALUEtransconvC/kernel7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEtransconvC/bias5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUE
 

�0
�1

�0
�1
�
�regularization_losses
 �layer_regularization_losses
�	variables
�metrics
�trainable_variables
�non_trainable_variables
�layers
�layer_metrics
 
 
 
�
�regularization_losses
 �layer_regularization_losses
�	variables
�metrics
�trainable_variables
�non_trainable_variables
�layers
�layer_metrics
\Z
VARIABLE_VALUEdecodeBa/kernel7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdecodeBa/bias5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUE
 

�0
�1

�0
�1
�
�regularization_losses
 �layer_regularization_losses
�	variables
�metrics
�trainable_variables
�non_trainable_variables
�layers
�layer_metrics
\Z
VARIABLE_VALUEdecodeBb/kernel7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdecodeBb/bias5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUE
 

�0
�1

�0
�1
�
�regularization_losses
 �layer_regularization_losses
�	variables
�metrics
�trainable_variables
�non_trainable_variables
�layers
�layer_metrics
^\
VARIABLE_VALUEtransconvB/kernel7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEtransconvB/bias5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUE
 

�0
�1

�0
�1
�
�regularization_losses
 �layer_regularization_losses
�	variables
�metrics
�trainable_variables
�non_trainable_variables
�layers
�layer_metrics
 
 
 
�
�regularization_losses
 �layer_regularization_losses
�	variables
�metrics
�trainable_variables
�non_trainable_variables
�layers
�layer_metrics
\Z
VARIABLE_VALUEdecodeAa/kernel7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdecodeAa/bias5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUE
 

�0
�1

�0
�1
�
�regularization_losses
 �layer_regularization_losses
�	variables
�metrics
�trainable_variables
�non_trainable_variables
�layers
�layer_metrics
\Z
VARIABLE_VALUEdecodeAb/kernel7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdecodeAb/bias5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUE
 

�0
�1

�0
�1
�
�regularization_losses
 �layer_regularization_losses
�	variables
�metrics
�trainable_variables
�non_trainable_variables
�layers
�layer_metrics
^\
VARIABLE_VALUEtransconvA/kernel7layer_with_weights-19/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEtransconvA/bias5layer_with_weights-19/bias/.ATTRIBUTES/VARIABLE_VALUE
 

�0
�1

�0
�1
�
�regularization_losses
 �layer_regularization_losses
�	variables
�metrics
�trainable_variables
�non_trainable_variables
�layers
�layer_metrics
 
 
 
�
�regularization_losses
 �layer_regularization_losses
�	variables
�metrics
�trainable_variables
�non_trainable_variables
�layers
�layer_metrics
\Z
VARIABLE_VALUEconvOuta/kernel7layer_with_weights-20/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconvOuta/bias5layer_with_weights-20/bias/.ATTRIBUTES/VARIABLE_VALUE
 

�0
�1

�0
�1
�
�regularization_losses
 �layer_regularization_losses
�	variables
�metrics
�trainable_variables
�non_trainable_variables
�layers
�layer_metrics
\Z
VARIABLE_VALUEconvOutb/kernel7layer_with_weights-21/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconvOutb/bias5layer_with_weights-21/bias/.ATTRIBUTES/VARIABLE_VALUE
 

�0
�1

�0
�1
�
�regularization_losses
 �layer_regularization_losses
�	variables
�metrics
�trainable_variables
�non_trainable_variables
�layers
�layer_metrics
b`
VARIABLE_VALUEPredictionMask/kernel7layer_with_weights-22/kernel/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEPredictionMask/bias5layer_with_weights-22/bias/.ATTRIBUTES/VARIABLE_VALUE
 

�0
�1

�0
�1
�
�regularization_losses
 �layer_regularization_losses
�	variables
�metrics
�trainable_variables
�non_trainable_variables
�layers
�layer_metrics
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 

�0
�1
�2
 
�
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
 31
!32
"33
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

�total

�count
�	variables
�	keras_api
I

�total

�count
�
_fn_kwargs
�	variables
�	keras_api
I

�total

�count
�
_fn_kwargs
�	variables
�	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

�0
�1

�	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

�0
�1

�	variables
QO
VARIABLE_VALUEtotal_24keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_24keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE
 

�0
�1

�	variables
~|
VARIABLE_VALUEAdam/encodeAa/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/encodeAa/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/encodeAb/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/encodeAb/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/encodeBa/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/encodeBa/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/encodeBb/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/encodeBb/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/encodeCa/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/encodeCa/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/encodeCb/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/encodeCb/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/encodeDa/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/encodeDa/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/encodeDb/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/encodeDb/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/encodeEa/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/encodeEa/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/encodeEb/kernel/mRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/encodeEb/bias/mPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
�
VARIABLE_VALUEAdam/transconvE/kernel/mSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/transconvE/bias/mQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/decodeCa/kernel/mSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/decodeCa/bias/mQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/decodeCb/kernel/mSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/decodeCb/bias/mQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
�
VARIABLE_VALUEAdam/transconvC/kernel/mSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/transconvC/bias/mQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/decodeBa/kernel/mSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/decodeBa/bias/mQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/decodeBb/kernel/mSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/decodeBb/bias/mQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
�
VARIABLE_VALUEAdam/transconvB/kernel/mSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/transconvB/bias/mQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/decodeAa/kernel/mSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/decodeAa/bias/mQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/decodeAb/kernel/mSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/decodeAb/bias/mQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
�
VARIABLE_VALUEAdam/transconvA/kernel/mSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/transconvA/bias/mQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/convOuta/kernel/mSlayer_with_weights-20/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/convOuta/bias/mQlayer_with_weights-20/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/convOutb/kernel/mSlayer_with_weights-21/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/convOutb/bias/mQlayer_with_weights-21/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEAdam/PredictionMask/kernel/mSlayer_with_weights-22/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
�
VARIABLE_VALUEAdam/PredictionMask/bias/mQlayer_with_weights-22/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/encodeAa/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/encodeAa/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/encodeAb/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/encodeAb/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/encodeBa/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/encodeBa/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/encodeBb/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/encodeBb/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/encodeCa/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/encodeCa/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/encodeCb/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/encodeCb/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/encodeDa/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/encodeDa/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/encodeDb/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/encodeDb/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/encodeEa/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/encodeEa/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/encodeEb/kernel/vRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/encodeEb/bias/vPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�
VARIABLE_VALUEAdam/transconvE/kernel/vSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/transconvE/bias/vQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/decodeCa/kernel/vSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/decodeCa/bias/vQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/decodeCb/kernel/vSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/decodeCb/bias/vQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�
VARIABLE_VALUEAdam/transconvC/kernel/vSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/transconvC/bias/vQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/decodeBa/kernel/vSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/decodeBa/bias/vQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/decodeBb/kernel/vSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/decodeBb/bias/vQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�
VARIABLE_VALUEAdam/transconvB/kernel/vSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/transconvB/bias/vQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/decodeAa/kernel/vSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/decodeAa/bias/vQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/decodeAb/kernel/vSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/decodeAb/bias/vQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�
VARIABLE_VALUEAdam/transconvA/kernel/vSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/transconvA/bias/vQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/convOuta/kernel/vSlayer_with_weights-20/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/convOuta/bias/vQlayer_with_weights-20/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/convOutb/kernel/vSlayer_with_weights-21/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/convOutb/bias/vQlayer_with_weights-21/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEAdam/PredictionMask/kernel/vSlayer_with_weights-22/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�
VARIABLE_VALUEAdam/PredictionMask/bias/vQlayer_with_weights-22/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�
serving_default_MRImagesPlaceholder*1
_output_shapes
:�����������*
dtype0*&
shape:�����������
�	
StatefulPartitionedCallStatefulPartitionedCallserving_default_MRImagesencodeAa/kernelencodeAa/biasencodeAb/kernelencodeAb/biasencodeBa/kernelencodeBa/biasencodeBb/kernelencodeBb/biasencodeCa/kernelencodeCa/biasencodeCb/kernelencodeCb/biasencodeDa/kernelencodeDa/biasencodeDb/kernelencodeDb/biasencodeEa/kernelencodeEa/biasencodeEb/kernelencodeEb/biastransconvE/kerneltransconvE/biasdecodeCa/kerneldecodeCa/biasdecodeCb/kerneldecodeCb/biastransconvC/kerneltransconvC/biasdecodeBa/kerneldecodeBa/biasdecodeBb/kerneldecodeBb/biastransconvB/kerneltransconvB/biasdecodeAa/kerneldecodeAa/biasdecodeAb/kerneldecodeAb/biastransconvA/kerneltransconvA/biasconvOuta/kernelconvOuta/biasconvOutb/kernelconvOutb/biasPredictionMask/kernelPredictionMask/bias*:
Tin3
12/*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*P
_read_only_resource_inputs2
0.	
 !"#$%&'()*+,-.*-
config_proto

CPU

GPU 2J 8� *,
f'R%
#__inference_signature_wrapper_19704
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�2
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#encodeAa/kernel/Read/ReadVariableOp!encodeAa/bias/Read/ReadVariableOp#encodeAb/kernel/Read/ReadVariableOp!encodeAb/bias/Read/ReadVariableOp#encodeBa/kernel/Read/ReadVariableOp!encodeBa/bias/Read/ReadVariableOp#encodeBb/kernel/Read/ReadVariableOp!encodeBb/bias/Read/ReadVariableOp#encodeCa/kernel/Read/ReadVariableOp!encodeCa/bias/Read/ReadVariableOp#encodeCb/kernel/Read/ReadVariableOp!encodeCb/bias/Read/ReadVariableOp#encodeDa/kernel/Read/ReadVariableOp!encodeDa/bias/Read/ReadVariableOp#encodeDb/kernel/Read/ReadVariableOp!encodeDb/bias/Read/ReadVariableOp#encodeEa/kernel/Read/ReadVariableOp!encodeEa/bias/Read/ReadVariableOp#encodeEb/kernel/Read/ReadVariableOp!encodeEb/bias/Read/ReadVariableOp%transconvE/kernel/Read/ReadVariableOp#transconvE/bias/Read/ReadVariableOp#decodeCa/kernel/Read/ReadVariableOp!decodeCa/bias/Read/ReadVariableOp#decodeCb/kernel/Read/ReadVariableOp!decodeCb/bias/Read/ReadVariableOp%transconvC/kernel/Read/ReadVariableOp#transconvC/bias/Read/ReadVariableOp#decodeBa/kernel/Read/ReadVariableOp!decodeBa/bias/Read/ReadVariableOp#decodeBb/kernel/Read/ReadVariableOp!decodeBb/bias/Read/ReadVariableOp%transconvB/kernel/Read/ReadVariableOp#transconvB/bias/Read/ReadVariableOp#decodeAa/kernel/Read/ReadVariableOp!decodeAa/bias/Read/ReadVariableOp#decodeAb/kernel/Read/ReadVariableOp!decodeAb/bias/Read/ReadVariableOp%transconvA/kernel/Read/ReadVariableOp#transconvA/bias/Read/ReadVariableOp#convOuta/kernel/Read/ReadVariableOp!convOuta/bias/Read/ReadVariableOp#convOutb/kernel/Read/ReadVariableOp!convOutb/bias/Read/ReadVariableOp)PredictionMask/kernel/Read/ReadVariableOp'PredictionMask/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOp*Adam/encodeAa/kernel/m/Read/ReadVariableOp(Adam/encodeAa/bias/m/Read/ReadVariableOp*Adam/encodeAb/kernel/m/Read/ReadVariableOp(Adam/encodeAb/bias/m/Read/ReadVariableOp*Adam/encodeBa/kernel/m/Read/ReadVariableOp(Adam/encodeBa/bias/m/Read/ReadVariableOp*Adam/encodeBb/kernel/m/Read/ReadVariableOp(Adam/encodeBb/bias/m/Read/ReadVariableOp*Adam/encodeCa/kernel/m/Read/ReadVariableOp(Adam/encodeCa/bias/m/Read/ReadVariableOp*Adam/encodeCb/kernel/m/Read/ReadVariableOp(Adam/encodeCb/bias/m/Read/ReadVariableOp*Adam/encodeDa/kernel/m/Read/ReadVariableOp(Adam/encodeDa/bias/m/Read/ReadVariableOp*Adam/encodeDb/kernel/m/Read/ReadVariableOp(Adam/encodeDb/bias/m/Read/ReadVariableOp*Adam/encodeEa/kernel/m/Read/ReadVariableOp(Adam/encodeEa/bias/m/Read/ReadVariableOp*Adam/encodeEb/kernel/m/Read/ReadVariableOp(Adam/encodeEb/bias/m/Read/ReadVariableOp,Adam/transconvE/kernel/m/Read/ReadVariableOp*Adam/transconvE/bias/m/Read/ReadVariableOp*Adam/decodeCa/kernel/m/Read/ReadVariableOp(Adam/decodeCa/bias/m/Read/ReadVariableOp*Adam/decodeCb/kernel/m/Read/ReadVariableOp(Adam/decodeCb/bias/m/Read/ReadVariableOp,Adam/transconvC/kernel/m/Read/ReadVariableOp*Adam/transconvC/bias/m/Read/ReadVariableOp*Adam/decodeBa/kernel/m/Read/ReadVariableOp(Adam/decodeBa/bias/m/Read/ReadVariableOp*Adam/decodeBb/kernel/m/Read/ReadVariableOp(Adam/decodeBb/bias/m/Read/ReadVariableOp,Adam/transconvB/kernel/m/Read/ReadVariableOp*Adam/transconvB/bias/m/Read/ReadVariableOp*Adam/decodeAa/kernel/m/Read/ReadVariableOp(Adam/decodeAa/bias/m/Read/ReadVariableOp*Adam/decodeAb/kernel/m/Read/ReadVariableOp(Adam/decodeAb/bias/m/Read/ReadVariableOp,Adam/transconvA/kernel/m/Read/ReadVariableOp*Adam/transconvA/bias/m/Read/ReadVariableOp*Adam/convOuta/kernel/m/Read/ReadVariableOp(Adam/convOuta/bias/m/Read/ReadVariableOp*Adam/convOutb/kernel/m/Read/ReadVariableOp(Adam/convOutb/bias/m/Read/ReadVariableOp0Adam/PredictionMask/kernel/m/Read/ReadVariableOp.Adam/PredictionMask/bias/m/Read/ReadVariableOp*Adam/encodeAa/kernel/v/Read/ReadVariableOp(Adam/encodeAa/bias/v/Read/ReadVariableOp*Adam/encodeAb/kernel/v/Read/ReadVariableOp(Adam/encodeAb/bias/v/Read/ReadVariableOp*Adam/encodeBa/kernel/v/Read/ReadVariableOp(Adam/encodeBa/bias/v/Read/ReadVariableOp*Adam/encodeBb/kernel/v/Read/ReadVariableOp(Adam/encodeBb/bias/v/Read/ReadVariableOp*Adam/encodeCa/kernel/v/Read/ReadVariableOp(Adam/encodeCa/bias/v/Read/ReadVariableOp*Adam/encodeCb/kernel/v/Read/ReadVariableOp(Adam/encodeCb/bias/v/Read/ReadVariableOp*Adam/encodeDa/kernel/v/Read/ReadVariableOp(Adam/encodeDa/bias/v/Read/ReadVariableOp*Adam/encodeDb/kernel/v/Read/ReadVariableOp(Adam/encodeDb/bias/v/Read/ReadVariableOp*Adam/encodeEa/kernel/v/Read/ReadVariableOp(Adam/encodeEa/bias/v/Read/ReadVariableOp*Adam/encodeEb/kernel/v/Read/ReadVariableOp(Adam/encodeEb/bias/v/Read/ReadVariableOp,Adam/transconvE/kernel/v/Read/ReadVariableOp*Adam/transconvE/bias/v/Read/ReadVariableOp*Adam/decodeCa/kernel/v/Read/ReadVariableOp(Adam/decodeCa/bias/v/Read/ReadVariableOp*Adam/decodeCb/kernel/v/Read/ReadVariableOp(Adam/decodeCb/bias/v/Read/ReadVariableOp,Adam/transconvC/kernel/v/Read/ReadVariableOp*Adam/transconvC/bias/v/Read/ReadVariableOp*Adam/decodeBa/kernel/v/Read/ReadVariableOp(Adam/decodeBa/bias/v/Read/ReadVariableOp*Adam/decodeBb/kernel/v/Read/ReadVariableOp(Adam/decodeBb/bias/v/Read/ReadVariableOp,Adam/transconvB/kernel/v/Read/ReadVariableOp*Adam/transconvB/bias/v/Read/ReadVariableOp*Adam/decodeAa/kernel/v/Read/ReadVariableOp(Adam/decodeAa/bias/v/Read/ReadVariableOp*Adam/decodeAb/kernel/v/Read/ReadVariableOp(Adam/decodeAb/bias/v/Read/ReadVariableOp,Adam/transconvA/kernel/v/Read/ReadVariableOp*Adam/transconvA/bias/v/Read/ReadVariableOp*Adam/convOuta/kernel/v/Read/ReadVariableOp(Adam/convOuta/bias/v/Read/ReadVariableOp*Adam/convOutb/kernel/v/Read/ReadVariableOp(Adam/convOutb/bias/v/Read/ReadVariableOp0Adam/PredictionMask/kernel/v/Read/ReadVariableOp.Adam/PredictionMask/bias/v/Read/ReadVariableOpConst*�
Tin�
�2�	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *'
f"R 
__inference__traced_save_21442
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameencodeAa/kernelencodeAa/biasencodeAb/kernelencodeAb/biasencodeBa/kernelencodeBa/biasencodeBb/kernelencodeBb/biasencodeCa/kernelencodeCa/biasencodeCb/kernelencodeCb/biasencodeDa/kernelencodeDa/biasencodeDb/kernelencodeDb/biasencodeEa/kernelencodeEa/biasencodeEb/kernelencodeEb/biastransconvE/kerneltransconvE/biasdecodeCa/kerneldecodeCa/biasdecodeCb/kerneldecodeCb/biastransconvC/kerneltransconvC/biasdecodeBa/kerneldecodeBa/biasdecodeBb/kerneldecodeBb/biastransconvB/kerneltransconvB/biasdecodeAa/kerneldecodeAa/biasdecodeAb/kerneldecodeAb/biastransconvA/kerneltransconvA/biasconvOuta/kernelconvOuta/biasconvOutb/kernelconvOutb/biasPredictionMask/kernelPredictionMask/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1total_2count_2Adam/encodeAa/kernel/mAdam/encodeAa/bias/mAdam/encodeAb/kernel/mAdam/encodeAb/bias/mAdam/encodeBa/kernel/mAdam/encodeBa/bias/mAdam/encodeBb/kernel/mAdam/encodeBb/bias/mAdam/encodeCa/kernel/mAdam/encodeCa/bias/mAdam/encodeCb/kernel/mAdam/encodeCb/bias/mAdam/encodeDa/kernel/mAdam/encodeDa/bias/mAdam/encodeDb/kernel/mAdam/encodeDb/bias/mAdam/encodeEa/kernel/mAdam/encodeEa/bias/mAdam/encodeEb/kernel/mAdam/encodeEb/bias/mAdam/transconvE/kernel/mAdam/transconvE/bias/mAdam/decodeCa/kernel/mAdam/decodeCa/bias/mAdam/decodeCb/kernel/mAdam/decodeCb/bias/mAdam/transconvC/kernel/mAdam/transconvC/bias/mAdam/decodeBa/kernel/mAdam/decodeBa/bias/mAdam/decodeBb/kernel/mAdam/decodeBb/bias/mAdam/transconvB/kernel/mAdam/transconvB/bias/mAdam/decodeAa/kernel/mAdam/decodeAa/bias/mAdam/decodeAb/kernel/mAdam/decodeAb/bias/mAdam/transconvA/kernel/mAdam/transconvA/bias/mAdam/convOuta/kernel/mAdam/convOuta/bias/mAdam/convOutb/kernel/mAdam/convOutb/bias/mAdam/PredictionMask/kernel/mAdam/PredictionMask/bias/mAdam/encodeAa/kernel/vAdam/encodeAa/bias/vAdam/encodeAb/kernel/vAdam/encodeAb/bias/vAdam/encodeBa/kernel/vAdam/encodeBa/bias/vAdam/encodeBb/kernel/vAdam/encodeBb/bias/vAdam/encodeCa/kernel/vAdam/encodeCa/bias/vAdam/encodeCb/kernel/vAdam/encodeCb/bias/vAdam/encodeDa/kernel/vAdam/encodeDa/bias/vAdam/encodeDb/kernel/vAdam/encodeDb/bias/vAdam/encodeEa/kernel/vAdam/encodeEa/bias/vAdam/encodeEb/kernel/vAdam/encodeEb/bias/vAdam/transconvE/kernel/vAdam/transconvE/bias/vAdam/decodeCa/kernel/vAdam/decodeCa/bias/vAdam/decodeCb/kernel/vAdam/decodeCb/bias/vAdam/transconvC/kernel/vAdam/transconvC/bias/vAdam/decodeBa/kernel/vAdam/decodeBa/bias/vAdam/decodeBb/kernel/vAdam/decodeBb/bias/vAdam/transconvB/kernel/vAdam/transconvB/bias/vAdam/decodeAa/kernel/vAdam/decodeAa/bias/vAdam/decodeAb/kernel/vAdam/decodeAb/bias/vAdam/transconvA/kernel/vAdam/transconvA/bias/vAdam/convOuta/kernel/vAdam/convOuta/bias/vAdam/convOutb/kernel/vAdam/convOutb/bias/vAdam/PredictionMask/kernel/vAdam/PredictionMask/bias/v*�
Tin�
�2�*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� **
f%R#
!__inference__traced_restore_21899��
�
k
L__inference_spatial_dropout2d_layer_call_and_return_conditional_losses_20511

inputs
identity�D
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dropout/Const�
dropout/MulMulinputsdropout/Const:output:0*
T0*J
_output_shapes8
6:4������������������������������������2
dropout/Mul�
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/1�
dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/2�
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0'dropout/random_uniform/shape/2:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
dropout/random_uniform/shape�
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*8
_output_shapes&
$:"������������������*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*8
_output_shapes&
$:"������������������2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*8
_output_shapes&
$:"������������������2
dropout/Cast�
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*J
_output_shapes8
6:4������������������������������������2
dropout/Mul_1�
IdentityIdentitydropout/Mul_1:z:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
}
(__inference_encodeCb_layer_call_fn_20584

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_encodeCb_layer_call_and_return_conditional_losses_185222
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������  @2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������  @::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������  @
 
_user_specified_nameinputs
�
S
'__inference_concatB_layer_call_fn_20859
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_concatB_layer_call_and_return_conditional_losses_188492
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������@@@2

Identity"
identityIdentity:output:0*[
_input_shapesJ
H:+��������������������������� :���������@@ :k g
A
_output_shapes/
-:+��������������������������� 
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:���������@@ 
"
_user_specified_name
inputs/1
�
}
(__inference_decodeAa_layer_call_fn_20879

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_decodeAa_layer_call_and_return_conditional_losses_188692
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������@@ 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������@@@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������@@@
 
_user_specified_nameinputs
�
}
(__inference_encodeBb_layer_call_fn_20468

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_encodeBb_layer_call_and_return_conditional_losses_184282
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������@@ 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������@@ ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������@@ 
 
_user_specified_nameinputs
�
}
(__inference_decodeBb_layer_call_fn_20846

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_decodeBb_layer_call_and_return_conditional_losses_188212
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������  @2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������  @::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������  @
 
_user_specified_nameinputs
�
\
@__inference_poolC_layer_call_and_return_conditional_losses_18069

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
2	
MaxPool�
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�

�
C__inference_decodeAb_layer_call_and_return_conditional_losses_18896

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@ *
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@ 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@@ 2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:���������@@ 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������@@ ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������@@ 
 
_user_specified_nameinputs
�
}
(__inference_encodeDa_layer_call_fn_20604

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_encodeDa_layer_call_and_return_conditional_losses_185502
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
��
�
Q__inference_2DUNet_Brats_Decathlon_layer_call_and_return_conditional_losses_19144
mrimages
encodeaa_19018
encodeaa_19020
encodeab_19023
encodeab_19025
encodeba_19029
encodeba_19031
encodebb_19034
encodebb_19036
encodeca_19040
encodeca_19042
encodecb_19046
encodecb_19048
encodeda_19052
encodeda_19054
encodedb_19058
encodedb_19060
encodeea_19064
encodeea_19066
encodeeb_19069
encodeeb_19071
transconve_19074
transconve_19076
decodeca_19080
decodeca_19082
decodecb_19085
decodecb_19087
transconvc_19090
transconvc_19092
decodeba_19096
decodeba_19098
decodebb_19101
decodebb_19103
transconvb_19106
transconvb_19108
decodeaa_19112
decodeaa_19114
decodeab_19117
decodeab_19119
transconva_19122
transconva_19124
convouta_19128
convouta_19130
convoutb_19133
convoutb_19135
predictionmask_19138
predictionmask_19140
identity��&PredictionMask/StatefulPartitionedCall� convOuta/StatefulPartitionedCall� convOutb/StatefulPartitionedCall� decodeAa/StatefulPartitionedCall� decodeAb/StatefulPartitionedCall� decodeBa/StatefulPartitionedCall� decodeBb/StatefulPartitionedCall� decodeCa/StatefulPartitionedCall� decodeCb/StatefulPartitionedCall� encodeAa/StatefulPartitionedCall� encodeAb/StatefulPartitionedCall� encodeBa/StatefulPartitionedCall� encodeBb/StatefulPartitionedCall� encodeCa/StatefulPartitionedCall� encodeCb/StatefulPartitionedCall� encodeDa/StatefulPartitionedCall� encodeDb/StatefulPartitionedCall� encodeEa/StatefulPartitionedCall� encodeEb/StatefulPartitionedCall�"transconvA/StatefulPartitionedCall�"transconvB/StatefulPartitionedCall�"transconvC/StatefulPartitionedCall�"transconvE/StatefulPartitionedCall�
 encodeAa/StatefulPartitionedCallStatefulPartitionedCallmrimagesencodeaa_19018encodeaa_19020*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_encodeAa_layer_call_and_return_conditional_losses_183462"
 encodeAa/StatefulPartitionedCall�
 encodeAb/StatefulPartitionedCallStatefulPartitionedCall)encodeAa/StatefulPartitionedCall:output:0encodeab_19023encodeab_19025*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_encodeAb_layer_call_and_return_conditional_losses_183732"
 encodeAb/StatefulPartitionedCall�
poolA/PartitionedCallPartitionedCall)encodeAb/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_poolA_layer_call_and_return_conditional_losses_179772
poolA/PartitionedCall�
 encodeBa/StatefulPartitionedCallStatefulPartitionedCallpoolA/PartitionedCall:output:0encodeba_19029encodeba_19031*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_encodeBa_layer_call_and_return_conditional_losses_184012"
 encodeBa/StatefulPartitionedCall�
 encodeBb/StatefulPartitionedCallStatefulPartitionedCall)encodeBa/StatefulPartitionedCall:output:0encodebb_19034encodebb_19036*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_encodeBb_layer_call_and_return_conditional_losses_184282"
 encodeBb/StatefulPartitionedCall�
poolB/PartitionedCallPartitionedCall)encodeBb/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������   * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_poolB_layer_call_and_return_conditional_losses_179892
poolB/PartitionedCall�
 encodeCa/StatefulPartitionedCallStatefulPartitionedCallpoolB/PartitionedCall:output:0encodeca_19040encodeca_19042*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_encodeCa_layer_call_and_return_conditional_losses_184562"
 encodeCa/StatefulPartitionedCall�
!spatial_dropout2d/PartitionedCallPartitionedCall)encodeCa/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_spatial_dropout2d_layer_call_and_return_conditional_losses_184992#
!spatial_dropout2d/PartitionedCall�
 encodeCb/StatefulPartitionedCallStatefulPartitionedCall*spatial_dropout2d/PartitionedCall:output:0encodecb_19046encodecb_19048*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_encodeCb_layer_call_and_return_conditional_losses_185222"
 encodeCb/StatefulPartitionedCall�
poolC/PartitionedCallPartitionedCall)encodeCb/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_poolC_layer_call_and_return_conditional_losses_180692
poolC/PartitionedCall�
 encodeDa/StatefulPartitionedCallStatefulPartitionedCallpoolC/PartitionedCall:output:0encodeda_19052encodeda_19054*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_encodeDa_layer_call_and_return_conditional_losses_185502"
 encodeDa/StatefulPartitionedCall�
#spatial_dropout2d_1/PartitionedCallPartitionedCall)encodeDa/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_spatial_dropout2d_1_layer_call_and_return_conditional_losses_185932%
#spatial_dropout2d_1/PartitionedCall�
 encodeDb/StatefulPartitionedCallStatefulPartitionedCall,spatial_dropout2d_1/PartitionedCall:output:0encodedb_19058encodedb_19060*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_encodeDb_layer_call_and_return_conditional_losses_186162"
 encodeDb/StatefulPartitionedCall�
poolD/PartitionedCallPartitionedCall)encodeDb/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_poolD_layer_call_and_return_conditional_losses_181492
poolD/PartitionedCall�
 encodeEa/StatefulPartitionedCallStatefulPartitionedCallpoolD/PartitionedCall:output:0encodeea_19064encodeea_19066*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_encodeEa_layer_call_and_return_conditional_losses_186442"
 encodeEa/StatefulPartitionedCall�
 encodeEb/StatefulPartitionedCallStatefulPartitionedCall)encodeEa/StatefulPartitionedCall:output:0encodeeb_19069encodeeb_19071*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_encodeEb_layer_call_and_return_conditional_losses_186712"
 encodeEb/StatefulPartitionedCall�
"transconvE/StatefulPartitionedCallStatefulPartitionedCall)encodeEb/StatefulPartitionedCall:output:0transconve_19074transconve_19076*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_transconvE_layer_call_and_return_conditional_losses_181892$
"transconvE/StatefulPartitionedCall�
concatD/PartitionedCallPartitionedCall+transconvE/StatefulPartitionedCall:output:0)encodeDb/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_concatD_layer_call_and_return_conditional_losses_186992
concatD/PartitionedCall�
 decodeCa/StatefulPartitionedCallStatefulPartitionedCall concatD/PartitionedCall:output:0decodeca_19080decodeca_19082*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_decodeCa_layer_call_and_return_conditional_losses_187192"
 decodeCa/StatefulPartitionedCall�
 decodeCb/StatefulPartitionedCallStatefulPartitionedCall)decodeCa/StatefulPartitionedCall:output:0decodecb_19085decodecb_19087*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_decodeCb_layer_call_and_return_conditional_losses_187462"
 decodeCb/StatefulPartitionedCall�
"transconvC/StatefulPartitionedCallStatefulPartitionedCall)decodeCb/StatefulPartitionedCall:output:0transconvc_19090transconvc_19092*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_transconvC_layer_call_and_return_conditional_losses_182332$
"transconvC/StatefulPartitionedCall�
concatC/PartitionedCallPartitionedCall+transconvC/StatefulPartitionedCall:output:0)encodeCb/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������  �* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_concatC_layer_call_and_return_conditional_losses_187742
concatC/PartitionedCall�
 decodeBa/StatefulPartitionedCallStatefulPartitionedCall concatC/PartitionedCall:output:0decodeba_19096decodeba_19098*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_decodeBa_layer_call_and_return_conditional_losses_187942"
 decodeBa/StatefulPartitionedCall�
 decodeBb/StatefulPartitionedCallStatefulPartitionedCall)decodeBa/StatefulPartitionedCall:output:0decodebb_19101decodebb_19103*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_decodeBb_layer_call_and_return_conditional_losses_188212"
 decodeBb/StatefulPartitionedCall�
"transconvB/StatefulPartitionedCallStatefulPartitionedCall)decodeBb/StatefulPartitionedCall:output:0transconvb_19106transconvb_19108*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+��������������������������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_transconvB_layer_call_and_return_conditional_losses_182772$
"transconvB/StatefulPartitionedCall�
concatB/PartitionedCallPartitionedCall+transconvB/StatefulPartitionedCall:output:0)encodeBb/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_concatB_layer_call_and_return_conditional_losses_188492
concatB/PartitionedCall�
 decodeAa/StatefulPartitionedCallStatefulPartitionedCall concatB/PartitionedCall:output:0decodeaa_19112decodeaa_19114*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_decodeAa_layer_call_and_return_conditional_losses_188692"
 decodeAa/StatefulPartitionedCall�
 decodeAb/StatefulPartitionedCallStatefulPartitionedCall)decodeAa/StatefulPartitionedCall:output:0decodeab_19117decodeab_19119*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_decodeAb_layer_call_and_return_conditional_losses_188962"
 decodeAb/StatefulPartitionedCall�
"transconvA/StatefulPartitionedCallStatefulPartitionedCall)decodeAb/StatefulPartitionedCall:output:0transconva_19122transconva_19124*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_transconvA_layer_call_and_return_conditional_losses_183212$
"transconvA/StatefulPartitionedCall�
concatA/PartitionedCallPartitionedCall+transconvA/StatefulPartitionedCall:output:0)encodeAb/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:����������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_concatA_layer_call_and_return_conditional_losses_189242
concatA/PartitionedCall�
 convOuta/StatefulPartitionedCallStatefulPartitionedCall concatA/PartitionedCall:output:0convouta_19128convouta_19130*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_convOuta_layer_call_and_return_conditional_losses_189442"
 convOuta/StatefulPartitionedCall�
 convOutb/StatefulPartitionedCallStatefulPartitionedCall)convOuta/StatefulPartitionedCall:output:0convoutb_19133convoutb_19135*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_convOutb_layer_call_and_return_conditional_losses_189712"
 convOutb/StatefulPartitionedCall�
&PredictionMask/StatefulPartitionedCallStatefulPartitionedCall)convOutb/StatefulPartitionedCall:output:0predictionmask_19138predictionmask_19140*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_PredictionMask_layer_call_and_return_conditional_losses_189982(
&PredictionMask/StatefulPartitionedCall�
IdentityIdentity/PredictionMask/StatefulPartitionedCall:output:0'^PredictionMask/StatefulPartitionedCall!^convOuta/StatefulPartitionedCall!^convOutb/StatefulPartitionedCall!^decodeAa/StatefulPartitionedCall!^decodeAb/StatefulPartitionedCall!^decodeBa/StatefulPartitionedCall!^decodeBb/StatefulPartitionedCall!^decodeCa/StatefulPartitionedCall!^decodeCb/StatefulPartitionedCall!^encodeAa/StatefulPartitionedCall!^encodeAb/StatefulPartitionedCall!^encodeBa/StatefulPartitionedCall!^encodeBb/StatefulPartitionedCall!^encodeCa/StatefulPartitionedCall!^encodeCb/StatefulPartitionedCall!^encodeDa/StatefulPartitionedCall!^encodeDb/StatefulPartitionedCall!^encodeEa/StatefulPartitionedCall!^encodeEb/StatefulPartitionedCall#^transconvA/StatefulPartitionedCall#^transconvB/StatefulPartitionedCall#^transconvC/StatefulPartitionedCall#^transconvE/StatefulPartitionedCall*
T0*1
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:�����������::::::::::::::::::::::::::::::::::::::::::::::2P
&PredictionMask/StatefulPartitionedCall&PredictionMask/StatefulPartitionedCall2D
 convOuta/StatefulPartitionedCall convOuta/StatefulPartitionedCall2D
 convOutb/StatefulPartitionedCall convOutb/StatefulPartitionedCall2D
 decodeAa/StatefulPartitionedCall decodeAa/StatefulPartitionedCall2D
 decodeAb/StatefulPartitionedCall decodeAb/StatefulPartitionedCall2D
 decodeBa/StatefulPartitionedCall decodeBa/StatefulPartitionedCall2D
 decodeBb/StatefulPartitionedCall decodeBb/StatefulPartitionedCall2D
 decodeCa/StatefulPartitionedCall decodeCa/StatefulPartitionedCall2D
 decodeCb/StatefulPartitionedCall decodeCb/StatefulPartitionedCall2D
 encodeAa/StatefulPartitionedCall encodeAa/StatefulPartitionedCall2D
 encodeAb/StatefulPartitionedCall encodeAb/StatefulPartitionedCall2D
 encodeBa/StatefulPartitionedCall encodeBa/StatefulPartitionedCall2D
 encodeBb/StatefulPartitionedCall encodeBb/StatefulPartitionedCall2D
 encodeCa/StatefulPartitionedCall encodeCa/StatefulPartitionedCall2D
 encodeCb/StatefulPartitionedCall encodeCb/StatefulPartitionedCall2D
 encodeDa/StatefulPartitionedCall encodeDa/StatefulPartitionedCall2D
 encodeDb/StatefulPartitionedCall encodeDb/StatefulPartitionedCall2D
 encodeEa/StatefulPartitionedCall encodeEa/StatefulPartitionedCall2D
 encodeEb/StatefulPartitionedCall encodeEb/StatefulPartitionedCall2H
"transconvA/StatefulPartitionedCall"transconvA/StatefulPartitionedCall2H
"transconvB/StatefulPartitionedCall"transconvB/StatefulPartitionedCall2H
"transconvC/StatefulPartitionedCall"transconvC/StatefulPartitionedCall2H
"transconvE/StatefulPartitionedCall"transconvE/StatefulPartitionedCall:[ W
1
_output_shapes
:�����������
"
_user_specified_name
MRImages
��
�<
__inference__traced_save_21442
file_prefix.
*savev2_encodeaa_kernel_read_readvariableop,
(savev2_encodeaa_bias_read_readvariableop.
*savev2_encodeab_kernel_read_readvariableop,
(savev2_encodeab_bias_read_readvariableop.
*savev2_encodeba_kernel_read_readvariableop,
(savev2_encodeba_bias_read_readvariableop.
*savev2_encodebb_kernel_read_readvariableop,
(savev2_encodebb_bias_read_readvariableop.
*savev2_encodeca_kernel_read_readvariableop,
(savev2_encodeca_bias_read_readvariableop.
*savev2_encodecb_kernel_read_readvariableop,
(savev2_encodecb_bias_read_readvariableop.
*savev2_encodeda_kernel_read_readvariableop,
(savev2_encodeda_bias_read_readvariableop.
*savev2_encodedb_kernel_read_readvariableop,
(savev2_encodedb_bias_read_readvariableop.
*savev2_encodeea_kernel_read_readvariableop,
(savev2_encodeea_bias_read_readvariableop.
*savev2_encodeeb_kernel_read_readvariableop,
(savev2_encodeeb_bias_read_readvariableop0
,savev2_transconve_kernel_read_readvariableop.
*savev2_transconve_bias_read_readvariableop.
*savev2_decodeca_kernel_read_readvariableop,
(savev2_decodeca_bias_read_readvariableop.
*savev2_decodecb_kernel_read_readvariableop,
(savev2_decodecb_bias_read_readvariableop0
,savev2_transconvc_kernel_read_readvariableop.
*savev2_transconvc_bias_read_readvariableop.
*savev2_decodeba_kernel_read_readvariableop,
(savev2_decodeba_bias_read_readvariableop.
*savev2_decodebb_kernel_read_readvariableop,
(savev2_decodebb_bias_read_readvariableop0
,savev2_transconvb_kernel_read_readvariableop.
*savev2_transconvb_bias_read_readvariableop.
*savev2_decodeaa_kernel_read_readvariableop,
(savev2_decodeaa_bias_read_readvariableop.
*savev2_decodeab_kernel_read_readvariableop,
(savev2_decodeab_bias_read_readvariableop0
,savev2_transconva_kernel_read_readvariableop.
*savev2_transconva_bias_read_readvariableop.
*savev2_convouta_kernel_read_readvariableop,
(savev2_convouta_bias_read_readvariableop.
*savev2_convoutb_kernel_read_readvariableop,
(savev2_convoutb_bias_read_readvariableop4
0savev2_predictionmask_kernel_read_readvariableop2
.savev2_predictionmask_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop&
"savev2_total_2_read_readvariableop&
"savev2_count_2_read_readvariableop5
1savev2_adam_encodeaa_kernel_m_read_readvariableop3
/savev2_adam_encodeaa_bias_m_read_readvariableop5
1savev2_adam_encodeab_kernel_m_read_readvariableop3
/savev2_adam_encodeab_bias_m_read_readvariableop5
1savev2_adam_encodeba_kernel_m_read_readvariableop3
/savev2_adam_encodeba_bias_m_read_readvariableop5
1savev2_adam_encodebb_kernel_m_read_readvariableop3
/savev2_adam_encodebb_bias_m_read_readvariableop5
1savev2_adam_encodeca_kernel_m_read_readvariableop3
/savev2_adam_encodeca_bias_m_read_readvariableop5
1savev2_adam_encodecb_kernel_m_read_readvariableop3
/savev2_adam_encodecb_bias_m_read_readvariableop5
1savev2_adam_encodeda_kernel_m_read_readvariableop3
/savev2_adam_encodeda_bias_m_read_readvariableop5
1savev2_adam_encodedb_kernel_m_read_readvariableop3
/savev2_adam_encodedb_bias_m_read_readvariableop5
1savev2_adam_encodeea_kernel_m_read_readvariableop3
/savev2_adam_encodeea_bias_m_read_readvariableop5
1savev2_adam_encodeeb_kernel_m_read_readvariableop3
/savev2_adam_encodeeb_bias_m_read_readvariableop7
3savev2_adam_transconve_kernel_m_read_readvariableop5
1savev2_adam_transconve_bias_m_read_readvariableop5
1savev2_adam_decodeca_kernel_m_read_readvariableop3
/savev2_adam_decodeca_bias_m_read_readvariableop5
1savev2_adam_decodecb_kernel_m_read_readvariableop3
/savev2_adam_decodecb_bias_m_read_readvariableop7
3savev2_adam_transconvc_kernel_m_read_readvariableop5
1savev2_adam_transconvc_bias_m_read_readvariableop5
1savev2_adam_decodeba_kernel_m_read_readvariableop3
/savev2_adam_decodeba_bias_m_read_readvariableop5
1savev2_adam_decodebb_kernel_m_read_readvariableop3
/savev2_adam_decodebb_bias_m_read_readvariableop7
3savev2_adam_transconvb_kernel_m_read_readvariableop5
1savev2_adam_transconvb_bias_m_read_readvariableop5
1savev2_adam_decodeaa_kernel_m_read_readvariableop3
/savev2_adam_decodeaa_bias_m_read_readvariableop5
1savev2_adam_decodeab_kernel_m_read_readvariableop3
/savev2_adam_decodeab_bias_m_read_readvariableop7
3savev2_adam_transconva_kernel_m_read_readvariableop5
1savev2_adam_transconva_bias_m_read_readvariableop5
1savev2_adam_convouta_kernel_m_read_readvariableop3
/savev2_adam_convouta_bias_m_read_readvariableop5
1savev2_adam_convoutb_kernel_m_read_readvariableop3
/savev2_adam_convoutb_bias_m_read_readvariableop;
7savev2_adam_predictionmask_kernel_m_read_readvariableop9
5savev2_adam_predictionmask_bias_m_read_readvariableop5
1savev2_adam_encodeaa_kernel_v_read_readvariableop3
/savev2_adam_encodeaa_bias_v_read_readvariableop5
1savev2_adam_encodeab_kernel_v_read_readvariableop3
/savev2_adam_encodeab_bias_v_read_readvariableop5
1savev2_adam_encodeba_kernel_v_read_readvariableop3
/savev2_adam_encodeba_bias_v_read_readvariableop5
1savev2_adam_encodebb_kernel_v_read_readvariableop3
/savev2_adam_encodebb_bias_v_read_readvariableop5
1savev2_adam_encodeca_kernel_v_read_readvariableop3
/savev2_adam_encodeca_bias_v_read_readvariableop5
1savev2_adam_encodecb_kernel_v_read_readvariableop3
/savev2_adam_encodecb_bias_v_read_readvariableop5
1savev2_adam_encodeda_kernel_v_read_readvariableop3
/savev2_adam_encodeda_bias_v_read_readvariableop5
1savev2_adam_encodedb_kernel_v_read_readvariableop3
/savev2_adam_encodedb_bias_v_read_readvariableop5
1savev2_adam_encodeea_kernel_v_read_readvariableop3
/savev2_adam_encodeea_bias_v_read_readvariableop5
1savev2_adam_encodeeb_kernel_v_read_readvariableop3
/savev2_adam_encodeeb_bias_v_read_readvariableop7
3savev2_adam_transconve_kernel_v_read_readvariableop5
1savev2_adam_transconve_bias_v_read_readvariableop5
1savev2_adam_decodeca_kernel_v_read_readvariableop3
/savev2_adam_decodeca_bias_v_read_readvariableop5
1savev2_adam_decodecb_kernel_v_read_readvariableop3
/savev2_adam_decodecb_bias_v_read_readvariableop7
3savev2_adam_transconvc_kernel_v_read_readvariableop5
1savev2_adam_transconvc_bias_v_read_readvariableop5
1savev2_adam_decodeba_kernel_v_read_readvariableop3
/savev2_adam_decodeba_bias_v_read_readvariableop5
1savev2_adam_decodebb_kernel_v_read_readvariableop3
/savev2_adam_decodebb_bias_v_read_readvariableop7
3savev2_adam_transconvb_kernel_v_read_readvariableop5
1savev2_adam_transconvb_bias_v_read_readvariableop5
1savev2_adam_decodeaa_kernel_v_read_readvariableop3
/savev2_adam_decodeaa_bias_v_read_readvariableop5
1savev2_adam_decodeab_kernel_v_read_readvariableop3
/savev2_adam_decodeab_bias_v_read_readvariableop7
3savev2_adam_transconva_kernel_v_read_readvariableop5
1savev2_adam_transconva_bias_v_read_readvariableop5
1savev2_adam_convouta_kernel_v_read_readvariableop3
/savev2_adam_convouta_bias_v_read_readvariableop5
1savev2_adam_convoutb_kernel_v_read_readvariableop3
/savev2_adam_convoutb_bias_v_read_readvariableop;
7savev2_adam_predictionmask_kernel_v_read_readvariableop9
5savev2_adam_predictionmask_bias_v_read_readvariableop
savev2_const

identity_1��MergeV2Checkpoints�
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�U
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes	
:�*
dtype0*�T
value�TB�T�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-19/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-19/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-20/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-20/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-21/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-21/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-22/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-22/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-20/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-20/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-21/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-21/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-22/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-22/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-20/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-20/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-21/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-21/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-22/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-22/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes	
:�*
dtype0*�
value�B��B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�9
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_encodeaa_kernel_read_readvariableop(savev2_encodeaa_bias_read_readvariableop*savev2_encodeab_kernel_read_readvariableop(savev2_encodeab_bias_read_readvariableop*savev2_encodeba_kernel_read_readvariableop(savev2_encodeba_bias_read_readvariableop*savev2_encodebb_kernel_read_readvariableop(savev2_encodebb_bias_read_readvariableop*savev2_encodeca_kernel_read_readvariableop(savev2_encodeca_bias_read_readvariableop*savev2_encodecb_kernel_read_readvariableop(savev2_encodecb_bias_read_readvariableop*savev2_encodeda_kernel_read_readvariableop(savev2_encodeda_bias_read_readvariableop*savev2_encodedb_kernel_read_readvariableop(savev2_encodedb_bias_read_readvariableop*savev2_encodeea_kernel_read_readvariableop(savev2_encodeea_bias_read_readvariableop*savev2_encodeeb_kernel_read_readvariableop(savev2_encodeeb_bias_read_readvariableop,savev2_transconve_kernel_read_readvariableop*savev2_transconve_bias_read_readvariableop*savev2_decodeca_kernel_read_readvariableop(savev2_decodeca_bias_read_readvariableop*savev2_decodecb_kernel_read_readvariableop(savev2_decodecb_bias_read_readvariableop,savev2_transconvc_kernel_read_readvariableop*savev2_transconvc_bias_read_readvariableop*savev2_decodeba_kernel_read_readvariableop(savev2_decodeba_bias_read_readvariableop*savev2_decodebb_kernel_read_readvariableop(savev2_decodebb_bias_read_readvariableop,savev2_transconvb_kernel_read_readvariableop*savev2_transconvb_bias_read_readvariableop*savev2_decodeaa_kernel_read_readvariableop(savev2_decodeaa_bias_read_readvariableop*savev2_decodeab_kernel_read_readvariableop(savev2_decodeab_bias_read_readvariableop,savev2_transconva_kernel_read_readvariableop*savev2_transconva_bias_read_readvariableop*savev2_convouta_kernel_read_readvariableop(savev2_convouta_bias_read_readvariableop*savev2_convoutb_kernel_read_readvariableop(savev2_convoutb_bias_read_readvariableop0savev2_predictionmask_kernel_read_readvariableop.savev2_predictionmask_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop1savev2_adam_encodeaa_kernel_m_read_readvariableop/savev2_adam_encodeaa_bias_m_read_readvariableop1savev2_adam_encodeab_kernel_m_read_readvariableop/savev2_adam_encodeab_bias_m_read_readvariableop1savev2_adam_encodeba_kernel_m_read_readvariableop/savev2_adam_encodeba_bias_m_read_readvariableop1savev2_adam_encodebb_kernel_m_read_readvariableop/savev2_adam_encodebb_bias_m_read_readvariableop1savev2_adam_encodeca_kernel_m_read_readvariableop/savev2_adam_encodeca_bias_m_read_readvariableop1savev2_adam_encodecb_kernel_m_read_readvariableop/savev2_adam_encodecb_bias_m_read_readvariableop1savev2_adam_encodeda_kernel_m_read_readvariableop/savev2_adam_encodeda_bias_m_read_readvariableop1savev2_adam_encodedb_kernel_m_read_readvariableop/savev2_adam_encodedb_bias_m_read_readvariableop1savev2_adam_encodeea_kernel_m_read_readvariableop/savev2_adam_encodeea_bias_m_read_readvariableop1savev2_adam_encodeeb_kernel_m_read_readvariableop/savev2_adam_encodeeb_bias_m_read_readvariableop3savev2_adam_transconve_kernel_m_read_readvariableop1savev2_adam_transconve_bias_m_read_readvariableop1savev2_adam_decodeca_kernel_m_read_readvariableop/savev2_adam_decodeca_bias_m_read_readvariableop1savev2_adam_decodecb_kernel_m_read_readvariableop/savev2_adam_decodecb_bias_m_read_readvariableop3savev2_adam_transconvc_kernel_m_read_readvariableop1savev2_adam_transconvc_bias_m_read_readvariableop1savev2_adam_decodeba_kernel_m_read_readvariableop/savev2_adam_decodeba_bias_m_read_readvariableop1savev2_adam_decodebb_kernel_m_read_readvariableop/savev2_adam_decodebb_bias_m_read_readvariableop3savev2_adam_transconvb_kernel_m_read_readvariableop1savev2_adam_transconvb_bias_m_read_readvariableop1savev2_adam_decodeaa_kernel_m_read_readvariableop/savev2_adam_decodeaa_bias_m_read_readvariableop1savev2_adam_decodeab_kernel_m_read_readvariableop/savev2_adam_decodeab_bias_m_read_readvariableop3savev2_adam_transconva_kernel_m_read_readvariableop1savev2_adam_transconva_bias_m_read_readvariableop1savev2_adam_convouta_kernel_m_read_readvariableop/savev2_adam_convouta_bias_m_read_readvariableop1savev2_adam_convoutb_kernel_m_read_readvariableop/savev2_adam_convoutb_bias_m_read_readvariableop7savev2_adam_predictionmask_kernel_m_read_readvariableop5savev2_adam_predictionmask_bias_m_read_readvariableop1savev2_adam_encodeaa_kernel_v_read_readvariableop/savev2_adam_encodeaa_bias_v_read_readvariableop1savev2_adam_encodeab_kernel_v_read_readvariableop/savev2_adam_encodeab_bias_v_read_readvariableop1savev2_adam_encodeba_kernel_v_read_readvariableop/savev2_adam_encodeba_bias_v_read_readvariableop1savev2_adam_encodebb_kernel_v_read_readvariableop/savev2_adam_encodebb_bias_v_read_readvariableop1savev2_adam_encodeca_kernel_v_read_readvariableop/savev2_adam_encodeca_bias_v_read_readvariableop1savev2_adam_encodecb_kernel_v_read_readvariableop/savev2_adam_encodecb_bias_v_read_readvariableop1savev2_adam_encodeda_kernel_v_read_readvariableop/savev2_adam_encodeda_bias_v_read_readvariableop1savev2_adam_encodedb_kernel_v_read_readvariableop/savev2_adam_encodedb_bias_v_read_readvariableop1savev2_adam_encodeea_kernel_v_read_readvariableop/savev2_adam_encodeea_bias_v_read_readvariableop1savev2_adam_encodeeb_kernel_v_read_readvariableop/savev2_adam_encodeeb_bias_v_read_readvariableop3savev2_adam_transconve_kernel_v_read_readvariableop1savev2_adam_transconve_bias_v_read_readvariableop1savev2_adam_decodeca_kernel_v_read_readvariableop/savev2_adam_decodeca_bias_v_read_readvariableop1savev2_adam_decodecb_kernel_v_read_readvariableop/savev2_adam_decodecb_bias_v_read_readvariableop3savev2_adam_transconvc_kernel_v_read_readvariableop1savev2_adam_transconvc_bias_v_read_readvariableop1savev2_adam_decodeba_kernel_v_read_readvariableop/savev2_adam_decodeba_bias_v_read_readvariableop1savev2_adam_decodebb_kernel_v_read_readvariableop/savev2_adam_decodebb_bias_v_read_readvariableop3savev2_adam_transconvb_kernel_v_read_readvariableop1savev2_adam_transconvb_bias_v_read_readvariableop1savev2_adam_decodeaa_kernel_v_read_readvariableop/savev2_adam_decodeaa_bias_v_read_readvariableop1savev2_adam_decodeab_kernel_v_read_readvariableop/savev2_adam_decodeab_bias_v_read_readvariableop3savev2_adam_transconva_kernel_v_read_readvariableop1savev2_adam_transconva_bias_v_read_readvariableop1savev2_adam_convouta_kernel_v_read_readvariableop/savev2_adam_convouta_bias_v_read_readvariableop1savev2_adam_convoutb_kernel_v_read_readvariableop/savev2_adam_convoutb_bias_v_read_readvariableop7savev2_adam_predictionmask_kernel_v_read_readvariableop5savev2_adam_predictionmask_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *�
dtypes�
�2�	2
SaveV2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*�
_input_shapes�
�: ::::: : :  : : @:@:@@:@:@�:�:��:�:��:�:��:�:��:�:��:�:��:�:@�:@:�@:@:@@:@: @: :@ : :  : : :: :::::: : : : : : : : : : : ::::: : :  : : @:@:@@:@:@�:�:��:�:��:�:��:�:��:�:��:�:��:�:@�:@:�@:@:@@:@: @: :@ : :  : : :: :::::::::: : :  : : @:@:@@:@:@�:�:��:�:��:�:��:�:��:�:��:�:��:�:@�:@:�@:@:@@:@: @: :@ : :  : : :: :::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: :,	(
&
_output_shapes
: @: 


_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:-)
'
_output_shapes
:@�:!

_output_shapes	
:�:.*
(
_output_shapes
:��:!

_output_shapes	
:�:.*
(
_output_shapes
:��:!

_output_shapes	
:�:.*
(
_output_shapes
:��:!

_output_shapes	
:�:.*
(
_output_shapes
:��:!

_output_shapes	
:�:.*
(
_output_shapes
:��:!

_output_shapes	
:�:.*
(
_output_shapes
:��:!

_output_shapes	
:�:-)
'
_output_shapes
:@�: 

_output_shapes
:@:-)
'
_output_shapes
:�@: 

_output_shapes
:@:,(
&
_output_shapes
:@@:  

_output_shapes
:@:,!(
&
_output_shapes
: @: "

_output_shapes
: :,#(
&
_output_shapes
:@ : $

_output_shapes
: :,%(
&
_output_shapes
:  : &

_output_shapes
: :,'(
&
_output_shapes
: : (

_output_shapes
::,)(
&
_output_shapes
: : *

_output_shapes
::,+(
&
_output_shapes
:: ,

_output_shapes
::,-(
&
_output_shapes
:: .

_output_shapes
::/

_output_shapes
: :0

_output_shapes
: :1

_output_shapes
: :2

_output_shapes
: :3

_output_shapes
: :4

_output_shapes
: :5

_output_shapes
: :6

_output_shapes
: :7

_output_shapes
: :8

_output_shapes
: :9

_output_shapes
: :,:(
&
_output_shapes
:: ;

_output_shapes
::,<(
&
_output_shapes
:: =

_output_shapes
::,>(
&
_output_shapes
: : ?

_output_shapes
: :,@(
&
_output_shapes
:  : A

_output_shapes
: :,B(
&
_output_shapes
: @: C

_output_shapes
:@:,D(
&
_output_shapes
:@@: E

_output_shapes
:@:-F)
'
_output_shapes
:@�:!G

_output_shapes	
:�:.H*
(
_output_shapes
:��:!I

_output_shapes	
:�:.J*
(
_output_shapes
:��:!K

_output_shapes	
:�:.L*
(
_output_shapes
:��:!M

_output_shapes	
:�:.N*
(
_output_shapes
:��:!O

_output_shapes	
:�:.P*
(
_output_shapes
:��:!Q

_output_shapes	
:�:.R*
(
_output_shapes
:��:!S

_output_shapes	
:�:-T)
'
_output_shapes
:@�: U

_output_shapes
:@:-V)
'
_output_shapes
:�@: W

_output_shapes
:@:,X(
&
_output_shapes
:@@: Y

_output_shapes
:@:,Z(
&
_output_shapes
: @: [

_output_shapes
: :,\(
&
_output_shapes
:@ : ]

_output_shapes
: :,^(
&
_output_shapes
:  : _

_output_shapes
: :,`(
&
_output_shapes
: : a

_output_shapes
::,b(
&
_output_shapes
: : c

_output_shapes
::,d(
&
_output_shapes
:: e

_output_shapes
::,f(
&
_output_shapes
:: g

_output_shapes
::,h(
&
_output_shapes
:: i

_output_shapes
::,j(
&
_output_shapes
:: k

_output_shapes
::,l(
&
_output_shapes
: : m

_output_shapes
: :,n(
&
_output_shapes
:  : o

_output_shapes
: :,p(
&
_output_shapes
: @: q

_output_shapes
:@:,r(
&
_output_shapes
:@@: s

_output_shapes
:@:-t)
'
_output_shapes
:@�:!u

_output_shapes	
:�:.v*
(
_output_shapes
:��:!w

_output_shapes	
:�:.x*
(
_output_shapes
:��:!y

_output_shapes	
:�:.z*
(
_output_shapes
:��:!{

_output_shapes	
:�:.|*
(
_output_shapes
:��:!}

_output_shapes	
:�:.~*
(
_output_shapes
:��:!

_output_shapes	
:�:/�*
(
_output_shapes
:��:"�

_output_shapes	
:�:.�)
'
_output_shapes
:@�:!�

_output_shapes
:@:.�)
'
_output_shapes
:�@:!�

_output_shapes
:@:-�(
&
_output_shapes
:@@:!�

_output_shapes
:@:-�(
&
_output_shapes
: @:!�

_output_shapes
: :-�(
&
_output_shapes
:@ :!�

_output_shapes
: :-�(
&
_output_shapes
:  :!�

_output_shapes
: :-�(
&
_output_shapes
: :!�

_output_shapes
::-�(
&
_output_shapes
: :!�

_output_shapes
::-�(
&
_output_shapes
::!�

_output_shapes
::-�(
&
_output_shapes
::!�

_output_shapes
::�

_output_shapes
: 
��
�
Q__inference_2DUNet_Brats_Decathlon_layer_call_and_return_conditional_losses_19502

inputs
encodeaa_19376
encodeaa_19378
encodeab_19381
encodeab_19383
encodeba_19387
encodeba_19389
encodebb_19392
encodebb_19394
encodeca_19398
encodeca_19400
encodecb_19404
encodecb_19406
encodeda_19410
encodeda_19412
encodedb_19416
encodedb_19418
encodeea_19422
encodeea_19424
encodeeb_19427
encodeeb_19429
transconve_19432
transconve_19434
decodeca_19438
decodeca_19440
decodecb_19443
decodecb_19445
transconvc_19448
transconvc_19450
decodeba_19454
decodeba_19456
decodebb_19459
decodebb_19461
transconvb_19464
transconvb_19466
decodeaa_19470
decodeaa_19472
decodeab_19475
decodeab_19477
transconva_19480
transconva_19482
convouta_19486
convouta_19488
convoutb_19491
convoutb_19493
predictionmask_19496
predictionmask_19498
identity��&PredictionMask/StatefulPartitionedCall� convOuta/StatefulPartitionedCall� convOutb/StatefulPartitionedCall� decodeAa/StatefulPartitionedCall� decodeAb/StatefulPartitionedCall� decodeBa/StatefulPartitionedCall� decodeBb/StatefulPartitionedCall� decodeCa/StatefulPartitionedCall� decodeCb/StatefulPartitionedCall� encodeAa/StatefulPartitionedCall� encodeAb/StatefulPartitionedCall� encodeBa/StatefulPartitionedCall� encodeBb/StatefulPartitionedCall� encodeCa/StatefulPartitionedCall� encodeCb/StatefulPartitionedCall� encodeDa/StatefulPartitionedCall� encodeDb/StatefulPartitionedCall� encodeEa/StatefulPartitionedCall� encodeEb/StatefulPartitionedCall�"transconvA/StatefulPartitionedCall�"transconvB/StatefulPartitionedCall�"transconvC/StatefulPartitionedCall�"transconvE/StatefulPartitionedCall�
 encodeAa/StatefulPartitionedCallStatefulPartitionedCallinputsencodeaa_19376encodeaa_19378*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_encodeAa_layer_call_and_return_conditional_losses_183462"
 encodeAa/StatefulPartitionedCall�
 encodeAb/StatefulPartitionedCallStatefulPartitionedCall)encodeAa/StatefulPartitionedCall:output:0encodeab_19381encodeab_19383*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_encodeAb_layer_call_and_return_conditional_losses_183732"
 encodeAb/StatefulPartitionedCall�
poolA/PartitionedCallPartitionedCall)encodeAb/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_poolA_layer_call_and_return_conditional_losses_179772
poolA/PartitionedCall�
 encodeBa/StatefulPartitionedCallStatefulPartitionedCallpoolA/PartitionedCall:output:0encodeba_19387encodeba_19389*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_encodeBa_layer_call_and_return_conditional_losses_184012"
 encodeBa/StatefulPartitionedCall�
 encodeBb/StatefulPartitionedCallStatefulPartitionedCall)encodeBa/StatefulPartitionedCall:output:0encodebb_19392encodebb_19394*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_encodeBb_layer_call_and_return_conditional_losses_184282"
 encodeBb/StatefulPartitionedCall�
poolB/PartitionedCallPartitionedCall)encodeBb/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������   * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_poolB_layer_call_and_return_conditional_losses_179892
poolB/PartitionedCall�
 encodeCa/StatefulPartitionedCallStatefulPartitionedCallpoolB/PartitionedCall:output:0encodeca_19398encodeca_19400*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_encodeCa_layer_call_and_return_conditional_losses_184562"
 encodeCa/StatefulPartitionedCall�
!spatial_dropout2d/PartitionedCallPartitionedCall)encodeCa/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_spatial_dropout2d_layer_call_and_return_conditional_losses_184992#
!spatial_dropout2d/PartitionedCall�
 encodeCb/StatefulPartitionedCallStatefulPartitionedCall*spatial_dropout2d/PartitionedCall:output:0encodecb_19404encodecb_19406*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_encodeCb_layer_call_and_return_conditional_losses_185222"
 encodeCb/StatefulPartitionedCall�
poolC/PartitionedCallPartitionedCall)encodeCb/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_poolC_layer_call_and_return_conditional_losses_180692
poolC/PartitionedCall�
 encodeDa/StatefulPartitionedCallStatefulPartitionedCallpoolC/PartitionedCall:output:0encodeda_19410encodeda_19412*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_encodeDa_layer_call_and_return_conditional_losses_185502"
 encodeDa/StatefulPartitionedCall�
#spatial_dropout2d_1/PartitionedCallPartitionedCall)encodeDa/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_spatial_dropout2d_1_layer_call_and_return_conditional_losses_185932%
#spatial_dropout2d_1/PartitionedCall�
 encodeDb/StatefulPartitionedCallStatefulPartitionedCall,spatial_dropout2d_1/PartitionedCall:output:0encodedb_19416encodedb_19418*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_encodeDb_layer_call_and_return_conditional_losses_186162"
 encodeDb/StatefulPartitionedCall�
poolD/PartitionedCallPartitionedCall)encodeDb/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_poolD_layer_call_and_return_conditional_losses_181492
poolD/PartitionedCall�
 encodeEa/StatefulPartitionedCallStatefulPartitionedCallpoolD/PartitionedCall:output:0encodeea_19422encodeea_19424*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_encodeEa_layer_call_and_return_conditional_losses_186442"
 encodeEa/StatefulPartitionedCall�
 encodeEb/StatefulPartitionedCallStatefulPartitionedCall)encodeEa/StatefulPartitionedCall:output:0encodeeb_19427encodeeb_19429*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_encodeEb_layer_call_and_return_conditional_losses_186712"
 encodeEb/StatefulPartitionedCall�
"transconvE/StatefulPartitionedCallStatefulPartitionedCall)encodeEb/StatefulPartitionedCall:output:0transconve_19432transconve_19434*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_transconvE_layer_call_and_return_conditional_losses_181892$
"transconvE/StatefulPartitionedCall�
concatD/PartitionedCallPartitionedCall+transconvE/StatefulPartitionedCall:output:0)encodeDb/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_concatD_layer_call_and_return_conditional_losses_186992
concatD/PartitionedCall�
 decodeCa/StatefulPartitionedCallStatefulPartitionedCall concatD/PartitionedCall:output:0decodeca_19438decodeca_19440*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_decodeCa_layer_call_and_return_conditional_losses_187192"
 decodeCa/StatefulPartitionedCall�
 decodeCb/StatefulPartitionedCallStatefulPartitionedCall)decodeCa/StatefulPartitionedCall:output:0decodecb_19443decodecb_19445*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_decodeCb_layer_call_and_return_conditional_losses_187462"
 decodeCb/StatefulPartitionedCall�
"transconvC/StatefulPartitionedCallStatefulPartitionedCall)decodeCb/StatefulPartitionedCall:output:0transconvc_19448transconvc_19450*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_transconvC_layer_call_and_return_conditional_losses_182332$
"transconvC/StatefulPartitionedCall�
concatC/PartitionedCallPartitionedCall+transconvC/StatefulPartitionedCall:output:0)encodeCb/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������  �* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_concatC_layer_call_and_return_conditional_losses_187742
concatC/PartitionedCall�
 decodeBa/StatefulPartitionedCallStatefulPartitionedCall concatC/PartitionedCall:output:0decodeba_19454decodeba_19456*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_decodeBa_layer_call_and_return_conditional_losses_187942"
 decodeBa/StatefulPartitionedCall�
 decodeBb/StatefulPartitionedCallStatefulPartitionedCall)decodeBa/StatefulPartitionedCall:output:0decodebb_19459decodebb_19461*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_decodeBb_layer_call_and_return_conditional_losses_188212"
 decodeBb/StatefulPartitionedCall�
"transconvB/StatefulPartitionedCallStatefulPartitionedCall)decodeBb/StatefulPartitionedCall:output:0transconvb_19464transconvb_19466*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+��������������������������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_transconvB_layer_call_and_return_conditional_losses_182772$
"transconvB/StatefulPartitionedCall�
concatB/PartitionedCallPartitionedCall+transconvB/StatefulPartitionedCall:output:0)encodeBb/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_concatB_layer_call_and_return_conditional_losses_188492
concatB/PartitionedCall�
 decodeAa/StatefulPartitionedCallStatefulPartitionedCall concatB/PartitionedCall:output:0decodeaa_19470decodeaa_19472*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_decodeAa_layer_call_and_return_conditional_losses_188692"
 decodeAa/StatefulPartitionedCall�
 decodeAb/StatefulPartitionedCallStatefulPartitionedCall)decodeAa/StatefulPartitionedCall:output:0decodeab_19475decodeab_19477*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_decodeAb_layer_call_and_return_conditional_losses_188962"
 decodeAb/StatefulPartitionedCall�
"transconvA/StatefulPartitionedCallStatefulPartitionedCall)decodeAb/StatefulPartitionedCall:output:0transconva_19480transconva_19482*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_transconvA_layer_call_and_return_conditional_losses_183212$
"transconvA/StatefulPartitionedCall�
concatA/PartitionedCallPartitionedCall+transconvA/StatefulPartitionedCall:output:0)encodeAb/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:����������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_concatA_layer_call_and_return_conditional_losses_189242
concatA/PartitionedCall�
 convOuta/StatefulPartitionedCallStatefulPartitionedCall concatA/PartitionedCall:output:0convouta_19486convouta_19488*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_convOuta_layer_call_and_return_conditional_losses_189442"
 convOuta/StatefulPartitionedCall�
 convOutb/StatefulPartitionedCallStatefulPartitionedCall)convOuta/StatefulPartitionedCall:output:0convoutb_19491convoutb_19493*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_convOutb_layer_call_and_return_conditional_losses_189712"
 convOutb/StatefulPartitionedCall�
&PredictionMask/StatefulPartitionedCallStatefulPartitionedCall)convOutb/StatefulPartitionedCall:output:0predictionmask_19496predictionmask_19498*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_PredictionMask_layer_call_and_return_conditional_losses_189982(
&PredictionMask/StatefulPartitionedCall�
IdentityIdentity/PredictionMask/StatefulPartitionedCall:output:0'^PredictionMask/StatefulPartitionedCall!^convOuta/StatefulPartitionedCall!^convOutb/StatefulPartitionedCall!^decodeAa/StatefulPartitionedCall!^decodeAb/StatefulPartitionedCall!^decodeBa/StatefulPartitionedCall!^decodeBb/StatefulPartitionedCall!^decodeCa/StatefulPartitionedCall!^decodeCb/StatefulPartitionedCall!^encodeAa/StatefulPartitionedCall!^encodeAb/StatefulPartitionedCall!^encodeBa/StatefulPartitionedCall!^encodeBb/StatefulPartitionedCall!^encodeCa/StatefulPartitionedCall!^encodeCb/StatefulPartitionedCall!^encodeDa/StatefulPartitionedCall!^encodeDb/StatefulPartitionedCall!^encodeEa/StatefulPartitionedCall!^encodeEb/StatefulPartitionedCall#^transconvA/StatefulPartitionedCall#^transconvB/StatefulPartitionedCall#^transconvC/StatefulPartitionedCall#^transconvE/StatefulPartitionedCall*
T0*1
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:�����������::::::::::::::::::::::::::::::::::::::::::::::2P
&PredictionMask/StatefulPartitionedCall&PredictionMask/StatefulPartitionedCall2D
 convOuta/StatefulPartitionedCall convOuta/StatefulPartitionedCall2D
 convOutb/StatefulPartitionedCall convOutb/StatefulPartitionedCall2D
 decodeAa/StatefulPartitionedCall decodeAa/StatefulPartitionedCall2D
 decodeAb/StatefulPartitionedCall decodeAb/StatefulPartitionedCall2D
 decodeBa/StatefulPartitionedCall decodeBa/StatefulPartitionedCall2D
 decodeBb/StatefulPartitionedCall decodeBb/StatefulPartitionedCall2D
 decodeCa/StatefulPartitionedCall decodeCa/StatefulPartitionedCall2D
 decodeCb/StatefulPartitionedCall decodeCb/StatefulPartitionedCall2D
 encodeAa/StatefulPartitionedCall encodeAa/StatefulPartitionedCall2D
 encodeAb/StatefulPartitionedCall encodeAb/StatefulPartitionedCall2D
 encodeBa/StatefulPartitionedCall encodeBa/StatefulPartitionedCall2D
 encodeBb/StatefulPartitionedCall encodeBb/StatefulPartitionedCall2D
 encodeCa/StatefulPartitionedCall encodeCa/StatefulPartitionedCall2D
 encodeCb/StatefulPartitionedCall encodeCb/StatefulPartitionedCall2D
 encodeDa/StatefulPartitionedCall encodeDa/StatefulPartitionedCall2D
 encodeDb/StatefulPartitionedCall encodeDb/StatefulPartitionedCall2D
 encodeEa/StatefulPartitionedCall encodeEa/StatefulPartitionedCall2D
 encodeEb/StatefulPartitionedCall encodeEb/StatefulPartitionedCall2H
"transconvA/StatefulPartitionedCall"transconvA/StatefulPartitionedCall2H
"transconvB/StatefulPartitionedCall"transconvB/StatefulPartitionedCall2H
"transconvC/StatefulPartitionedCall"transconvC/StatefulPartitionedCall2H
"transconvE/StatefulPartitionedCall"transconvE/StatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�

�
C__inference_convOutb_layer_call_and_return_conditional_losses_18971

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:�����������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:�����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�

�
C__inference_encodeEb_layer_call_and_return_conditional_losses_20731

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:����������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
C__inference_encodeAb_layer_call_and_return_conditional_losses_18373

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:�����������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:�����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�

�
C__inference_decodeAa_layer_call_and_return_conditional_losses_18869

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@ *
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@ 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@@ 2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:���������@@ 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������@@@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������@@@
 
_user_specified_nameinputs
�
k
L__inference_spatial_dropout2d_layer_call_and_return_conditional_losses_18050

inputs
identity�D
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dropout/Const�
dropout/MulMulinputsdropout/Const:output:0*
T0*J
_output_shapes8
6:4������������������������������������2
dropout/Mul�
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/1�
dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/2�
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0'dropout/random_uniform/shape/2:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
dropout/random_uniform/shape�
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*8
_output_shapes&
$:"������������������*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*8
_output_shapes&
$:"������������������2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*8
_output_shapes&
$:"������������������2
dropout/Cast�
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*J
_output_shapes8
6:4������������������������������������2
dropout/Mul_1�
IdentityIdentitydropout/Mul_1:z:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�

�
C__inference_convOuta_layer_call_and_return_conditional_losses_20923

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:�����������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:����������� ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:����������� 
 
_user_specified_nameinputs
�#
�
E__inference_transconvC_layer_call_and_return_conditional_losses_18233

inputs,
(conv2d_transpose_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2	
stack/3�
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*'
_output_shapes
:@�*
dtype02!
conv2d_transpose/ReadVariableOp�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+���������������������������@*
paddingSAME*
strides
2
conv2d_transpose�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������@2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,����������������������������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
6__inference_2DUNet_Brats_Decathlon_layer_call_fn_20291

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44*:
Tin3
12/*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*P
_read_only_resource_inputs2
0.	
 !"#$%&'()*+,-.*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_2DUNet_Brats_Decathlon_layer_call_and_return_conditional_losses_192762
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:�����������::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�

�
C__inference_decodeAa_layer_call_and_return_conditional_losses_20870

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@ *
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@ 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@@ 2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:���������@@ 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������@@@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������@@@
 
_user_specified_nameinputs
�
A
%__inference_poolC_layer_call_fn_18075

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_poolC_layer_call_and_return_conditional_losses_180692
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�

�
C__inference_encodeEb_layer_call_and_return_conditional_losses_18671

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:����������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
C__inference_encodeCa_layer_call_and_return_conditional_losses_20479

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  @*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  @2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������  @2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:���������  @2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������   ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������   
 
_user_specified_nameinputs
�

�
C__inference_encodeDa_layer_call_and_return_conditional_losses_20595

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:����������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�#
�
E__inference_transconvE_layer_call_and_return_conditional_losses_18189

inputs,
(conv2d_transpose_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1U
stack/3Const*
_output_shapes
: *
dtype0*
value
B :�2	
stack/3�
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*(
_output_shapes
:��*
dtype02!
conv2d_transpose/ReadVariableOp�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*B
_output_shapes0
.:,����������������������������*
paddingSAME*
strides
2
conv2d_transpose�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,����������������������������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�

�
C__inference_encodeEa_layer_call_and_return_conditional_losses_18644

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:����������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
l
B__inference_concatC_layer_call_and_return_conditional_losses_18774

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis�
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*0
_output_shapes
:���������  �2
concatl
IdentityIdentityconcat:output:0*
T0*0
_output_shapes
:���������  �2

Identity"
identityIdentity:output:0*[
_input_shapesJ
H:+���������������������������@:���������  @:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs:WS
/
_output_shapes
:���������  @
 
_user_specified_nameinputs
�
\
@__inference_poolD_layer_call_and_return_conditional_losses_18149

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
2	
MaxPool�
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
6__inference_2DUNet_Brats_Decathlon_layer_call_fn_20388

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44*:
Tin3
12/*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*P
_read_only_resource_inputs2
0.	
 !"#$%&'()*+,-.*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_2DUNet_Brats_Decathlon_layer_call_and_return_conditional_losses_195022
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:�����������::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�

*__inference_transconvA_layer_call_fn_18331

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_transconvA_layer_call_and_return_conditional_losses_183212
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+��������������������������� ::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
A
%__inference_poolB_layer_call_fn_17995

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_poolB_layer_call_and_return_conditional_losses_179892
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�

�
I__inference_PredictionMask_layer_call_and_return_conditional_losses_20963

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������*
paddingVALID*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������2	
BiasAddk
SigmoidSigmoidBiasAdd:output:0*
T0*1
_output_shapes
:�����������2	
Sigmoid�
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:�����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
M
1__inference_spatial_dropout2d_layer_call_fn_20526

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_spatial_dropout2d_layer_call_and_return_conditional_losses_180602
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
m
N__inference_spatial_dropout2d_1_layer_call_and_return_conditional_losses_18130

inputs
identity�D
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dropout/Const�
dropout/MulMulinputsdropout/Const:output:0*
T0*J
_output_shapes8
6:4������������������������������������2
dropout/Mul�
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/1�
dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/2�
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0'dropout/random_uniform/shape/2:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
dropout/random_uniform/shape�
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*8
_output_shapes&
$:"������������������*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*8
_output_shapes&
$:"������������������2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*8
_output_shapes&
$:"������������������2
dropout/Cast�
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*J
_output_shapes8
6:4������������������������������������2
dropout/Mul_1�
IdentityIdentitydropout/Mul_1:z:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
m
N__inference_spatial_dropout2d_1_layer_call_and_return_conditional_losses_20627

inputs
identity�D
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dropout/Const|
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:����������2
dropout/Mul�
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/1�
dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/2�
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0'dropout/random_uniform/shape/2:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
dropout/random_uniform/shape�
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*8
_output_shapes&
$:"������������������*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*8
_output_shapes&
$:"������������������2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*8
_output_shapes&
$:"������������������2
dropout/Cast�
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:����������2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
m
N__inference_spatial_dropout2d_1_layer_call_and_return_conditional_losses_18588

inputs
identity�D
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dropout/Const|
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:����������2
dropout/Mul�
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/1�
dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/2�
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0'dropout/random_uniform/shape/2:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
dropout/random_uniform/shape�
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*8
_output_shapes&
$:"������������������*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*8
_output_shapes&
$:"������������������2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*8
_output_shapes&
$:"������������������2
dropout/Cast�
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:����������2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
j
L__inference_spatial_dropout2d_layer_call_and_return_conditional_losses_20554

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:���������  @2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:���������  @2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:���������  @:W S
/
_output_shapes
:���������  @
 
_user_specified_nameinputs
�

�
C__inference_encodeAa_layer_call_and_return_conditional_losses_18346

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:�����������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:�����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
\
@__inference_poolB_layer_call_and_return_conditional_losses_17989

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
2	
MaxPool�
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�

�
C__inference_decodeBa_layer_call_and_return_conditional_losses_18794

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:�@*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  @*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  @2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������  @2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:���������  @2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:���������  �::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:���������  �
 
_user_specified_nameinputs
�#
�
E__inference_transconvB_layer_call_and_return_conditional_losses_18277

inputs,
(conv2d_transpose_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2	
stack/3�
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv2d_transpose/ReadVariableOp�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+��������������������������� *
paddingSAME*
strides
2
conv2d_transpose�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+��������������������������� 2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
l
3__inference_spatial_dropout2d_1_layer_call_fn_20675

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_spatial_dropout2d_1_layer_call_and_return_conditional_losses_181302
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������22
StatefulPartitionedCallStatefulPartitionedCall:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�

�
C__inference_encodeDb_layer_call_and_return_conditional_losses_20691

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:����������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
}
(__inference_encodeAa_layer_call_fn_20408

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_encodeAa_layer_call_and_return_conditional_losses_183462
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:�����������::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�

�
C__inference_encodeCa_layer_call_and_return_conditional_losses_18456

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  @*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  @2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������  @2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:���������  @2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������   ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������   
 
_user_specified_nameinputs
�
l
B__inference_concatB_layer_call_and_return_conditional_losses_18849

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis�
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*/
_output_shapes
:���������@@@2
concatk
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:���������@@@2

Identity"
identityIdentity:output:0*[
_input_shapesJ
H:+��������������������������� :���������@@ :i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs:WS
/
_output_shapes
:���������@@ 
 
_user_specified_nameinputs
�
k
L__inference_spatial_dropout2d_layer_call_and_return_conditional_losses_18494

inputs
identity�D
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:���������  @2
dropout/Mul�
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/1�
dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/2�
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0'dropout/random_uniform/shape/2:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
dropout/random_uniform/shape�
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*8
_output_shapes&
$:"������������������*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*8
_output_shapes&
$:"������������������2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*8
_output_shapes&
$:"������������������2
dropout/Cast�
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:���������  @2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:���������  @2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������  @:W S
/
_output_shapes
:���������  @
 
_user_specified_nameinputs
�
}
(__inference_convOuta_layer_call_fn_20932

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_convOuta_layer_call_and_return_conditional_losses_189442
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:����������� ::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:����������� 
 
_user_specified_nameinputs
�
}
(__inference_encodeDb_layer_call_fn_20700

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_encodeDb_layer_call_and_return_conditional_losses_186162
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
��
�N
!__inference__traced_restore_21899
file_prefix$
 assignvariableop_encodeaa_kernel$
 assignvariableop_1_encodeaa_bias&
"assignvariableop_2_encodeab_kernel$
 assignvariableop_3_encodeab_bias&
"assignvariableop_4_encodeba_kernel$
 assignvariableop_5_encodeba_bias&
"assignvariableop_6_encodebb_kernel$
 assignvariableop_7_encodebb_bias&
"assignvariableop_8_encodeca_kernel$
 assignvariableop_9_encodeca_bias'
#assignvariableop_10_encodecb_kernel%
!assignvariableop_11_encodecb_bias'
#assignvariableop_12_encodeda_kernel%
!assignvariableop_13_encodeda_bias'
#assignvariableop_14_encodedb_kernel%
!assignvariableop_15_encodedb_bias'
#assignvariableop_16_encodeea_kernel%
!assignvariableop_17_encodeea_bias'
#assignvariableop_18_encodeeb_kernel%
!assignvariableop_19_encodeeb_bias)
%assignvariableop_20_transconve_kernel'
#assignvariableop_21_transconve_bias'
#assignvariableop_22_decodeca_kernel%
!assignvariableop_23_decodeca_bias'
#assignvariableop_24_decodecb_kernel%
!assignvariableop_25_decodecb_bias)
%assignvariableop_26_transconvc_kernel'
#assignvariableop_27_transconvc_bias'
#assignvariableop_28_decodeba_kernel%
!assignvariableop_29_decodeba_bias'
#assignvariableop_30_decodebb_kernel%
!assignvariableop_31_decodebb_bias)
%assignvariableop_32_transconvb_kernel'
#assignvariableop_33_transconvb_bias'
#assignvariableop_34_decodeaa_kernel%
!assignvariableop_35_decodeaa_bias'
#assignvariableop_36_decodeab_kernel%
!assignvariableop_37_decodeab_bias)
%assignvariableop_38_transconva_kernel'
#assignvariableop_39_transconva_bias'
#assignvariableop_40_convouta_kernel%
!assignvariableop_41_convouta_bias'
#assignvariableop_42_convoutb_kernel%
!assignvariableop_43_convoutb_bias-
)assignvariableop_44_predictionmask_kernel+
'assignvariableop_45_predictionmask_bias!
assignvariableop_46_adam_iter#
assignvariableop_47_adam_beta_1#
assignvariableop_48_adam_beta_2"
assignvariableop_49_adam_decay*
&assignvariableop_50_adam_learning_rate
assignvariableop_51_total
assignvariableop_52_count
assignvariableop_53_total_1
assignvariableop_54_count_1
assignvariableop_55_total_2
assignvariableop_56_count_2.
*assignvariableop_57_adam_encodeaa_kernel_m,
(assignvariableop_58_adam_encodeaa_bias_m.
*assignvariableop_59_adam_encodeab_kernel_m,
(assignvariableop_60_adam_encodeab_bias_m.
*assignvariableop_61_adam_encodeba_kernel_m,
(assignvariableop_62_adam_encodeba_bias_m.
*assignvariableop_63_adam_encodebb_kernel_m,
(assignvariableop_64_adam_encodebb_bias_m.
*assignvariableop_65_adam_encodeca_kernel_m,
(assignvariableop_66_adam_encodeca_bias_m.
*assignvariableop_67_adam_encodecb_kernel_m,
(assignvariableop_68_adam_encodecb_bias_m.
*assignvariableop_69_adam_encodeda_kernel_m,
(assignvariableop_70_adam_encodeda_bias_m.
*assignvariableop_71_adam_encodedb_kernel_m,
(assignvariableop_72_adam_encodedb_bias_m.
*assignvariableop_73_adam_encodeea_kernel_m,
(assignvariableop_74_adam_encodeea_bias_m.
*assignvariableop_75_adam_encodeeb_kernel_m,
(assignvariableop_76_adam_encodeeb_bias_m0
,assignvariableop_77_adam_transconve_kernel_m.
*assignvariableop_78_adam_transconve_bias_m.
*assignvariableop_79_adam_decodeca_kernel_m,
(assignvariableop_80_adam_decodeca_bias_m.
*assignvariableop_81_adam_decodecb_kernel_m,
(assignvariableop_82_adam_decodecb_bias_m0
,assignvariableop_83_adam_transconvc_kernel_m.
*assignvariableop_84_adam_transconvc_bias_m.
*assignvariableop_85_adam_decodeba_kernel_m,
(assignvariableop_86_adam_decodeba_bias_m.
*assignvariableop_87_adam_decodebb_kernel_m,
(assignvariableop_88_adam_decodebb_bias_m0
,assignvariableop_89_adam_transconvb_kernel_m.
*assignvariableop_90_adam_transconvb_bias_m.
*assignvariableop_91_adam_decodeaa_kernel_m,
(assignvariableop_92_adam_decodeaa_bias_m.
*assignvariableop_93_adam_decodeab_kernel_m,
(assignvariableop_94_adam_decodeab_bias_m0
,assignvariableop_95_adam_transconva_kernel_m.
*assignvariableop_96_adam_transconva_bias_m.
*assignvariableop_97_adam_convouta_kernel_m,
(assignvariableop_98_adam_convouta_bias_m.
*assignvariableop_99_adam_convoutb_kernel_m-
)assignvariableop_100_adam_convoutb_bias_m5
1assignvariableop_101_adam_predictionmask_kernel_m3
/assignvariableop_102_adam_predictionmask_bias_m/
+assignvariableop_103_adam_encodeaa_kernel_v-
)assignvariableop_104_adam_encodeaa_bias_v/
+assignvariableop_105_adam_encodeab_kernel_v-
)assignvariableop_106_adam_encodeab_bias_v/
+assignvariableop_107_adam_encodeba_kernel_v-
)assignvariableop_108_adam_encodeba_bias_v/
+assignvariableop_109_adam_encodebb_kernel_v-
)assignvariableop_110_adam_encodebb_bias_v/
+assignvariableop_111_adam_encodeca_kernel_v-
)assignvariableop_112_adam_encodeca_bias_v/
+assignvariableop_113_adam_encodecb_kernel_v-
)assignvariableop_114_adam_encodecb_bias_v/
+assignvariableop_115_adam_encodeda_kernel_v-
)assignvariableop_116_adam_encodeda_bias_v/
+assignvariableop_117_adam_encodedb_kernel_v-
)assignvariableop_118_adam_encodedb_bias_v/
+assignvariableop_119_adam_encodeea_kernel_v-
)assignvariableop_120_adam_encodeea_bias_v/
+assignvariableop_121_adam_encodeeb_kernel_v-
)assignvariableop_122_adam_encodeeb_bias_v1
-assignvariableop_123_adam_transconve_kernel_v/
+assignvariableop_124_adam_transconve_bias_v/
+assignvariableop_125_adam_decodeca_kernel_v-
)assignvariableop_126_adam_decodeca_bias_v/
+assignvariableop_127_adam_decodecb_kernel_v-
)assignvariableop_128_adam_decodecb_bias_v1
-assignvariableop_129_adam_transconvc_kernel_v/
+assignvariableop_130_adam_transconvc_bias_v/
+assignvariableop_131_adam_decodeba_kernel_v-
)assignvariableop_132_adam_decodeba_bias_v/
+assignvariableop_133_adam_decodebb_kernel_v-
)assignvariableop_134_adam_decodebb_bias_v1
-assignvariableop_135_adam_transconvb_kernel_v/
+assignvariableop_136_adam_transconvb_bias_v/
+assignvariableop_137_adam_decodeaa_kernel_v-
)assignvariableop_138_adam_decodeaa_bias_v/
+assignvariableop_139_adam_decodeab_kernel_v-
)assignvariableop_140_adam_decodeab_bias_v1
-assignvariableop_141_adam_transconva_kernel_v/
+assignvariableop_142_adam_transconva_bias_v/
+assignvariableop_143_adam_convouta_kernel_v-
)assignvariableop_144_adam_convouta_bias_v/
+assignvariableop_145_adam_convoutb_kernel_v-
)assignvariableop_146_adam_convoutb_bias_v5
1assignvariableop_147_adam_predictionmask_kernel_v3
/assignvariableop_148_adam_predictionmask_bias_v
identity_150��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_100�AssignVariableOp_101�AssignVariableOp_102�AssignVariableOp_103�AssignVariableOp_104�AssignVariableOp_105�AssignVariableOp_106�AssignVariableOp_107�AssignVariableOp_108�AssignVariableOp_109�AssignVariableOp_11�AssignVariableOp_110�AssignVariableOp_111�AssignVariableOp_112�AssignVariableOp_113�AssignVariableOp_114�AssignVariableOp_115�AssignVariableOp_116�AssignVariableOp_117�AssignVariableOp_118�AssignVariableOp_119�AssignVariableOp_12�AssignVariableOp_120�AssignVariableOp_121�AssignVariableOp_122�AssignVariableOp_123�AssignVariableOp_124�AssignVariableOp_125�AssignVariableOp_126�AssignVariableOp_127�AssignVariableOp_128�AssignVariableOp_129�AssignVariableOp_13�AssignVariableOp_130�AssignVariableOp_131�AssignVariableOp_132�AssignVariableOp_133�AssignVariableOp_134�AssignVariableOp_135�AssignVariableOp_136�AssignVariableOp_137�AssignVariableOp_138�AssignVariableOp_139�AssignVariableOp_14�AssignVariableOp_140�AssignVariableOp_141�AssignVariableOp_142�AssignVariableOp_143�AssignVariableOp_144�AssignVariableOp_145�AssignVariableOp_146�AssignVariableOp_147�AssignVariableOp_148�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_49�AssignVariableOp_5�AssignVariableOp_50�AssignVariableOp_51�AssignVariableOp_52�AssignVariableOp_53�AssignVariableOp_54�AssignVariableOp_55�AssignVariableOp_56�AssignVariableOp_57�AssignVariableOp_58�AssignVariableOp_59�AssignVariableOp_6�AssignVariableOp_60�AssignVariableOp_61�AssignVariableOp_62�AssignVariableOp_63�AssignVariableOp_64�AssignVariableOp_65�AssignVariableOp_66�AssignVariableOp_67�AssignVariableOp_68�AssignVariableOp_69�AssignVariableOp_7�AssignVariableOp_70�AssignVariableOp_71�AssignVariableOp_72�AssignVariableOp_73�AssignVariableOp_74�AssignVariableOp_75�AssignVariableOp_76�AssignVariableOp_77�AssignVariableOp_78�AssignVariableOp_79�AssignVariableOp_8�AssignVariableOp_80�AssignVariableOp_81�AssignVariableOp_82�AssignVariableOp_83�AssignVariableOp_84�AssignVariableOp_85�AssignVariableOp_86�AssignVariableOp_87�AssignVariableOp_88�AssignVariableOp_89�AssignVariableOp_9�AssignVariableOp_90�AssignVariableOp_91�AssignVariableOp_92�AssignVariableOp_93�AssignVariableOp_94�AssignVariableOp_95�AssignVariableOp_96�AssignVariableOp_97�AssignVariableOp_98�AssignVariableOp_99�U
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes	
:�*
dtype0*�T
value�TB�T�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-19/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-19/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-20/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-20/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-21/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-21/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-22/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-22/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-20/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-20/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-21/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-21/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-22/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-22/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-20/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-20/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-21/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-21/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-22/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-22/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes	
:�*
dtype0*�
value�B��B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*�
dtypes�
�2�	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOp assignvariableop_encodeaa_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOp assignvariableop_1_encodeaa_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOp"assignvariableop_2_encodeab_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOp assignvariableop_3_encodeab_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOp"assignvariableop_4_encodeba_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOp assignvariableop_5_encodeba_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOp"assignvariableop_6_encodebb_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOp assignvariableop_7_encodebb_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOp"assignvariableop_8_encodeca_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOp assignvariableop_9_encodeca_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOp#assignvariableop_10_encodecb_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOp!assignvariableop_11_encodecb_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOp#assignvariableop_12_encodeda_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOp!assignvariableop_13_encodeda_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOp#assignvariableop_14_encodedb_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOp!assignvariableop_15_encodedb_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOp#assignvariableop_16_encodeea_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOp!assignvariableop_17_encodeea_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOp#assignvariableop_18_encodeeb_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19�
AssignVariableOp_19AssignVariableOp!assignvariableop_19_encodeeb_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20�
AssignVariableOp_20AssignVariableOp%assignvariableop_20_transconve_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21�
AssignVariableOp_21AssignVariableOp#assignvariableop_21_transconve_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22�
AssignVariableOp_22AssignVariableOp#assignvariableop_22_decodeca_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23�
AssignVariableOp_23AssignVariableOp!assignvariableop_23_decodeca_biasIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24�
AssignVariableOp_24AssignVariableOp#assignvariableop_24_decodecb_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25�
AssignVariableOp_25AssignVariableOp!assignvariableop_25_decodecb_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26�
AssignVariableOp_26AssignVariableOp%assignvariableop_26_transconvc_kernelIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27�
AssignVariableOp_27AssignVariableOp#assignvariableop_27_transconvc_biasIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28�
AssignVariableOp_28AssignVariableOp#assignvariableop_28_decodeba_kernelIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29�
AssignVariableOp_29AssignVariableOp!assignvariableop_29_decodeba_biasIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30�
AssignVariableOp_30AssignVariableOp#assignvariableop_30_decodebb_kernelIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31�
AssignVariableOp_31AssignVariableOp!assignvariableop_31_decodebb_biasIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32�
AssignVariableOp_32AssignVariableOp%assignvariableop_32_transconvb_kernelIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33�
AssignVariableOp_33AssignVariableOp#assignvariableop_33_transconvb_biasIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34�
AssignVariableOp_34AssignVariableOp#assignvariableop_34_decodeaa_kernelIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35�
AssignVariableOp_35AssignVariableOp!assignvariableop_35_decodeaa_biasIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36�
AssignVariableOp_36AssignVariableOp#assignvariableop_36_decodeab_kernelIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37�
AssignVariableOp_37AssignVariableOp!assignvariableop_37_decodeab_biasIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38�
AssignVariableOp_38AssignVariableOp%assignvariableop_38_transconva_kernelIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39�
AssignVariableOp_39AssignVariableOp#assignvariableop_39_transconva_biasIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40�
AssignVariableOp_40AssignVariableOp#assignvariableop_40_convouta_kernelIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41�
AssignVariableOp_41AssignVariableOp!assignvariableop_41_convouta_biasIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42�
AssignVariableOp_42AssignVariableOp#assignvariableop_42_convoutb_kernelIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43�
AssignVariableOp_43AssignVariableOp!assignvariableop_43_convoutb_biasIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44�
AssignVariableOp_44AssignVariableOp)assignvariableop_44_predictionmask_kernelIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45�
AssignVariableOp_45AssignVariableOp'assignvariableop_45_predictionmask_biasIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_46�
AssignVariableOp_46AssignVariableOpassignvariableop_46_adam_iterIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47�
AssignVariableOp_47AssignVariableOpassignvariableop_47_adam_beta_1Identity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48�
AssignVariableOp_48AssignVariableOpassignvariableop_48_adam_beta_2Identity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49�
AssignVariableOp_49AssignVariableOpassignvariableop_49_adam_decayIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50�
AssignVariableOp_50AssignVariableOp&assignvariableop_50_adam_learning_rateIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51�
AssignVariableOp_51AssignVariableOpassignvariableop_51_totalIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52�
AssignVariableOp_52AssignVariableOpassignvariableop_52_countIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53�
AssignVariableOp_53AssignVariableOpassignvariableop_53_total_1Identity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54�
AssignVariableOp_54AssignVariableOpassignvariableop_54_count_1Identity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55�
AssignVariableOp_55AssignVariableOpassignvariableop_55_total_2Identity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56�
AssignVariableOp_56AssignVariableOpassignvariableop_56_count_2Identity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57�
AssignVariableOp_57AssignVariableOp*assignvariableop_57_adam_encodeaa_kernel_mIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58�
AssignVariableOp_58AssignVariableOp(assignvariableop_58_adam_encodeaa_bias_mIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59�
AssignVariableOp_59AssignVariableOp*assignvariableop_59_adam_encodeab_kernel_mIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60�
AssignVariableOp_60AssignVariableOp(assignvariableop_60_adam_encodeab_bias_mIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61�
AssignVariableOp_61AssignVariableOp*assignvariableop_61_adam_encodeba_kernel_mIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62�
AssignVariableOp_62AssignVariableOp(assignvariableop_62_adam_encodeba_bias_mIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63�
AssignVariableOp_63AssignVariableOp*assignvariableop_63_adam_encodebb_kernel_mIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64�
AssignVariableOp_64AssignVariableOp(assignvariableop_64_adam_encodebb_bias_mIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65�
AssignVariableOp_65AssignVariableOp*assignvariableop_65_adam_encodeca_kernel_mIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66�
AssignVariableOp_66AssignVariableOp(assignvariableop_66_adam_encodeca_bias_mIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67�
AssignVariableOp_67AssignVariableOp*assignvariableop_67_adam_encodecb_kernel_mIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68�
AssignVariableOp_68AssignVariableOp(assignvariableop_68_adam_encodecb_bias_mIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69�
AssignVariableOp_69AssignVariableOp*assignvariableop_69_adam_encodeda_kernel_mIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70�
AssignVariableOp_70AssignVariableOp(assignvariableop_70_adam_encodeda_bias_mIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71�
AssignVariableOp_71AssignVariableOp*assignvariableop_71_adam_encodedb_kernel_mIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72�
AssignVariableOp_72AssignVariableOp(assignvariableop_72_adam_encodedb_bias_mIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_72n
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:2
Identity_73�
AssignVariableOp_73AssignVariableOp*assignvariableop_73_adam_encodeea_kernel_mIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_73n
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:2
Identity_74�
AssignVariableOp_74AssignVariableOp(assignvariableop_74_adam_encodeea_bias_mIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_74n
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:2
Identity_75�
AssignVariableOp_75AssignVariableOp*assignvariableop_75_adam_encodeeb_kernel_mIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_75n
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:2
Identity_76�
AssignVariableOp_76AssignVariableOp(assignvariableop_76_adam_encodeeb_bias_mIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_76n
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:2
Identity_77�
AssignVariableOp_77AssignVariableOp,assignvariableop_77_adam_transconve_kernel_mIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_77n
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:2
Identity_78�
AssignVariableOp_78AssignVariableOp*assignvariableop_78_adam_transconve_bias_mIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_78n
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:2
Identity_79�
AssignVariableOp_79AssignVariableOp*assignvariableop_79_adam_decodeca_kernel_mIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79n
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:2
Identity_80�
AssignVariableOp_80AssignVariableOp(assignvariableop_80_adam_decodeca_bias_mIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_80n
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:2
Identity_81�
AssignVariableOp_81AssignVariableOp*assignvariableop_81_adam_decodecb_kernel_mIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_81n
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:2
Identity_82�
AssignVariableOp_82AssignVariableOp(assignvariableop_82_adam_decodecb_bias_mIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_82n
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:2
Identity_83�
AssignVariableOp_83AssignVariableOp,assignvariableop_83_adam_transconvc_kernel_mIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_83n
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:2
Identity_84�
AssignVariableOp_84AssignVariableOp*assignvariableop_84_adam_transconvc_bias_mIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_84n
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:2
Identity_85�
AssignVariableOp_85AssignVariableOp*assignvariableop_85_adam_decodeba_kernel_mIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_85n
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:2
Identity_86�
AssignVariableOp_86AssignVariableOp(assignvariableop_86_adam_decodeba_bias_mIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_86n
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:2
Identity_87�
AssignVariableOp_87AssignVariableOp*assignvariableop_87_adam_decodebb_kernel_mIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_87n
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:2
Identity_88�
AssignVariableOp_88AssignVariableOp(assignvariableop_88_adam_decodebb_bias_mIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_88n
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:2
Identity_89�
AssignVariableOp_89AssignVariableOp,assignvariableop_89_adam_transconvb_kernel_mIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_89n
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:2
Identity_90�
AssignVariableOp_90AssignVariableOp*assignvariableop_90_adam_transconvb_bias_mIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_90n
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:2
Identity_91�
AssignVariableOp_91AssignVariableOp*assignvariableop_91_adam_decodeaa_kernel_mIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_91n
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:2
Identity_92�
AssignVariableOp_92AssignVariableOp(assignvariableop_92_adam_decodeaa_bias_mIdentity_92:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_92n
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:2
Identity_93�
AssignVariableOp_93AssignVariableOp*assignvariableop_93_adam_decodeab_kernel_mIdentity_93:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_93n
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:2
Identity_94�
AssignVariableOp_94AssignVariableOp(assignvariableop_94_adam_decodeab_bias_mIdentity_94:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_94n
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:2
Identity_95�
AssignVariableOp_95AssignVariableOp,assignvariableop_95_adam_transconva_kernel_mIdentity_95:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_95n
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:2
Identity_96�
AssignVariableOp_96AssignVariableOp*assignvariableop_96_adam_transconva_bias_mIdentity_96:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_96n
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:2
Identity_97�
AssignVariableOp_97AssignVariableOp*assignvariableop_97_adam_convouta_kernel_mIdentity_97:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_97n
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:2
Identity_98�
AssignVariableOp_98AssignVariableOp(assignvariableop_98_adam_convouta_bias_mIdentity_98:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_98n
Identity_99IdentityRestoreV2:tensors:99"/device:CPU:0*
T0*
_output_shapes
:2
Identity_99�
AssignVariableOp_99AssignVariableOp*assignvariableop_99_adam_convoutb_kernel_mIdentity_99:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_99q
Identity_100IdentityRestoreV2:tensors:100"/device:CPU:0*
T0*
_output_shapes
:2
Identity_100�
AssignVariableOp_100AssignVariableOp)assignvariableop_100_adam_convoutb_bias_mIdentity_100:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_100q
Identity_101IdentityRestoreV2:tensors:101"/device:CPU:0*
T0*
_output_shapes
:2
Identity_101�
AssignVariableOp_101AssignVariableOp1assignvariableop_101_adam_predictionmask_kernel_mIdentity_101:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_101q
Identity_102IdentityRestoreV2:tensors:102"/device:CPU:0*
T0*
_output_shapes
:2
Identity_102�
AssignVariableOp_102AssignVariableOp/assignvariableop_102_adam_predictionmask_bias_mIdentity_102:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_102q
Identity_103IdentityRestoreV2:tensors:103"/device:CPU:0*
T0*
_output_shapes
:2
Identity_103�
AssignVariableOp_103AssignVariableOp+assignvariableop_103_adam_encodeaa_kernel_vIdentity_103:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_103q
Identity_104IdentityRestoreV2:tensors:104"/device:CPU:0*
T0*
_output_shapes
:2
Identity_104�
AssignVariableOp_104AssignVariableOp)assignvariableop_104_adam_encodeaa_bias_vIdentity_104:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_104q
Identity_105IdentityRestoreV2:tensors:105"/device:CPU:0*
T0*
_output_shapes
:2
Identity_105�
AssignVariableOp_105AssignVariableOp+assignvariableop_105_adam_encodeab_kernel_vIdentity_105:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_105q
Identity_106IdentityRestoreV2:tensors:106"/device:CPU:0*
T0*
_output_shapes
:2
Identity_106�
AssignVariableOp_106AssignVariableOp)assignvariableop_106_adam_encodeab_bias_vIdentity_106:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_106q
Identity_107IdentityRestoreV2:tensors:107"/device:CPU:0*
T0*
_output_shapes
:2
Identity_107�
AssignVariableOp_107AssignVariableOp+assignvariableop_107_adam_encodeba_kernel_vIdentity_107:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_107q
Identity_108IdentityRestoreV2:tensors:108"/device:CPU:0*
T0*
_output_shapes
:2
Identity_108�
AssignVariableOp_108AssignVariableOp)assignvariableop_108_adam_encodeba_bias_vIdentity_108:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_108q
Identity_109IdentityRestoreV2:tensors:109"/device:CPU:0*
T0*
_output_shapes
:2
Identity_109�
AssignVariableOp_109AssignVariableOp+assignvariableop_109_adam_encodebb_kernel_vIdentity_109:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_109q
Identity_110IdentityRestoreV2:tensors:110"/device:CPU:0*
T0*
_output_shapes
:2
Identity_110�
AssignVariableOp_110AssignVariableOp)assignvariableop_110_adam_encodebb_bias_vIdentity_110:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_110q
Identity_111IdentityRestoreV2:tensors:111"/device:CPU:0*
T0*
_output_shapes
:2
Identity_111�
AssignVariableOp_111AssignVariableOp+assignvariableop_111_adam_encodeca_kernel_vIdentity_111:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_111q
Identity_112IdentityRestoreV2:tensors:112"/device:CPU:0*
T0*
_output_shapes
:2
Identity_112�
AssignVariableOp_112AssignVariableOp)assignvariableop_112_adam_encodeca_bias_vIdentity_112:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_112q
Identity_113IdentityRestoreV2:tensors:113"/device:CPU:0*
T0*
_output_shapes
:2
Identity_113�
AssignVariableOp_113AssignVariableOp+assignvariableop_113_adam_encodecb_kernel_vIdentity_113:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_113q
Identity_114IdentityRestoreV2:tensors:114"/device:CPU:0*
T0*
_output_shapes
:2
Identity_114�
AssignVariableOp_114AssignVariableOp)assignvariableop_114_adam_encodecb_bias_vIdentity_114:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_114q
Identity_115IdentityRestoreV2:tensors:115"/device:CPU:0*
T0*
_output_shapes
:2
Identity_115�
AssignVariableOp_115AssignVariableOp+assignvariableop_115_adam_encodeda_kernel_vIdentity_115:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_115q
Identity_116IdentityRestoreV2:tensors:116"/device:CPU:0*
T0*
_output_shapes
:2
Identity_116�
AssignVariableOp_116AssignVariableOp)assignvariableop_116_adam_encodeda_bias_vIdentity_116:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_116q
Identity_117IdentityRestoreV2:tensors:117"/device:CPU:0*
T0*
_output_shapes
:2
Identity_117�
AssignVariableOp_117AssignVariableOp+assignvariableop_117_adam_encodedb_kernel_vIdentity_117:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_117q
Identity_118IdentityRestoreV2:tensors:118"/device:CPU:0*
T0*
_output_shapes
:2
Identity_118�
AssignVariableOp_118AssignVariableOp)assignvariableop_118_adam_encodedb_bias_vIdentity_118:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_118q
Identity_119IdentityRestoreV2:tensors:119"/device:CPU:0*
T0*
_output_shapes
:2
Identity_119�
AssignVariableOp_119AssignVariableOp+assignvariableop_119_adam_encodeea_kernel_vIdentity_119:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_119q
Identity_120IdentityRestoreV2:tensors:120"/device:CPU:0*
T0*
_output_shapes
:2
Identity_120�
AssignVariableOp_120AssignVariableOp)assignvariableop_120_adam_encodeea_bias_vIdentity_120:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_120q
Identity_121IdentityRestoreV2:tensors:121"/device:CPU:0*
T0*
_output_shapes
:2
Identity_121�
AssignVariableOp_121AssignVariableOp+assignvariableop_121_adam_encodeeb_kernel_vIdentity_121:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_121q
Identity_122IdentityRestoreV2:tensors:122"/device:CPU:0*
T0*
_output_shapes
:2
Identity_122�
AssignVariableOp_122AssignVariableOp)assignvariableop_122_adam_encodeeb_bias_vIdentity_122:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_122q
Identity_123IdentityRestoreV2:tensors:123"/device:CPU:0*
T0*
_output_shapes
:2
Identity_123�
AssignVariableOp_123AssignVariableOp-assignvariableop_123_adam_transconve_kernel_vIdentity_123:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_123q
Identity_124IdentityRestoreV2:tensors:124"/device:CPU:0*
T0*
_output_shapes
:2
Identity_124�
AssignVariableOp_124AssignVariableOp+assignvariableop_124_adam_transconve_bias_vIdentity_124:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_124q
Identity_125IdentityRestoreV2:tensors:125"/device:CPU:0*
T0*
_output_shapes
:2
Identity_125�
AssignVariableOp_125AssignVariableOp+assignvariableop_125_adam_decodeca_kernel_vIdentity_125:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_125q
Identity_126IdentityRestoreV2:tensors:126"/device:CPU:0*
T0*
_output_shapes
:2
Identity_126�
AssignVariableOp_126AssignVariableOp)assignvariableop_126_adam_decodeca_bias_vIdentity_126:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_126q
Identity_127IdentityRestoreV2:tensors:127"/device:CPU:0*
T0*
_output_shapes
:2
Identity_127�
AssignVariableOp_127AssignVariableOp+assignvariableop_127_adam_decodecb_kernel_vIdentity_127:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_127q
Identity_128IdentityRestoreV2:tensors:128"/device:CPU:0*
T0*
_output_shapes
:2
Identity_128�
AssignVariableOp_128AssignVariableOp)assignvariableop_128_adam_decodecb_bias_vIdentity_128:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_128q
Identity_129IdentityRestoreV2:tensors:129"/device:CPU:0*
T0*
_output_shapes
:2
Identity_129�
AssignVariableOp_129AssignVariableOp-assignvariableop_129_adam_transconvc_kernel_vIdentity_129:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_129q
Identity_130IdentityRestoreV2:tensors:130"/device:CPU:0*
T0*
_output_shapes
:2
Identity_130�
AssignVariableOp_130AssignVariableOp+assignvariableop_130_adam_transconvc_bias_vIdentity_130:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_130q
Identity_131IdentityRestoreV2:tensors:131"/device:CPU:0*
T0*
_output_shapes
:2
Identity_131�
AssignVariableOp_131AssignVariableOp+assignvariableop_131_adam_decodeba_kernel_vIdentity_131:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_131q
Identity_132IdentityRestoreV2:tensors:132"/device:CPU:0*
T0*
_output_shapes
:2
Identity_132�
AssignVariableOp_132AssignVariableOp)assignvariableop_132_adam_decodeba_bias_vIdentity_132:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_132q
Identity_133IdentityRestoreV2:tensors:133"/device:CPU:0*
T0*
_output_shapes
:2
Identity_133�
AssignVariableOp_133AssignVariableOp+assignvariableop_133_adam_decodebb_kernel_vIdentity_133:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_133q
Identity_134IdentityRestoreV2:tensors:134"/device:CPU:0*
T0*
_output_shapes
:2
Identity_134�
AssignVariableOp_134AssignVariableOp)assignvariableop_134_adam_decodebb_bias_vIdentity_134:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_134q
Identity_135IdentityRestoreV2:tensors:135"/device:CPU:0*
T0*
_output_shapes
:2
Identity_135�
AssignVariableOp_135AssignVariableOp-assignvariableop_135_adam_transconvb_kernel_vIdentity_135:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_135q
Identity_136IdentityRestoreV2:tensors:136"/device:CPU:0*
T0*
_output_shapes
:2
Identity_136�
AssignVariableOp_136AssignVariableOp+assignvariableop_136_adam_transconvb_bias_vIdentity_136:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_136q
Identity_137IdentityRestoreV2:tensors:137"/device:CPU:0*
T0*
_output_shapes
:2
Identity_137�
AssignVariableOp_137AssignVariableOp+assignvariableop_137_adam_decodeaa_kernel_vIdentity_137:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_137q
Identity_138IdentityRestoreV2:tensors:138"/device:CPU:0*
T0*
_output_shapes
:2
Identity_138�
AssignVariableOp_138AssignVariableOp)assignvariableop_138_adam_decodeaa_bias_vIdentity_138:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_138q
Identity_139IdentityRestoreV2:tensors:139"/device:CPU:0*
T0*
_output_shapes
:2
Identity_139�
AssignVariableOp_139AssignVariableOp+assignvariableop_139_adam_decodeab_kernel_vIdentity_139:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_139q
Identity_140IdentityRestoreV2:tensors:140"/device:CPU:0*
T0*
_output_shapes
:2
Identity_140�
AssignVariableOp_140AssignVariableOp)assignvariableop_140_adam_decodeab_bias_vIdentity_140:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_140q
Identity_141IdentityRestoreV2:tensors:141"/device:CPU:0*
T0*
_output_shapes
:2
Identity_141�
AssignVariableOp_141AssignVariableOp-assignvariableop_141_adam_transconva_kernel_vIdentity_141:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_141q
Identity_142IdentityRestoreV2:tensors:142"/device:CPU:0*
T0*
_output_shapes
:2
Identity_142�
AssignVariableOp_142AssignVariableOp+assignvariableop_142_adam_transconva_bias_vIdentity_142:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_142q
Identity_143IdentityRestoreV2:tensors:143"/device:CPU:0*
T0*
_output_shapes
:2
Identity_143�
AssignVariableOp_143AssignVariableOp+assignvariableop_143_adam_convouta_kernel_vIdentity_143:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_143q
Identity_144IdentityRestoreV2:tensors:144"/device:CPU:0*
T0*
_output_shapes
:2
Identity_144�
AssignVariableOp_144AssignVariableOp)assignvariableop_144_adam_convouta_bias_vIdentity_144:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_144q
Identity_145IdentityRestoreV2:tensors:145"/device:CPU:0*
T0*
_output_shapes
:2
Identity_145�
AssignVariableOp_145AssignVariableOp+assignvariableop_145_adam_convoutb_kernel_vIdentity_145:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_145q
Identity_146IdentityRestoreV2:tensors:146"/device:CPU:0*
T0*
_output_shapes
:2
Identity_146�
AssignVariableOp_146AssignVariableOp)assignvariableop_146_adam_convoutb_bias_vIdentity_146:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_146q
Identity_147IdentityRestoreV2:tensors:147"/device:CPU:0*
T0*
_output_shapes
:2
Identity_147�
AssignVariableOp_147AssignVariableOp1assignvariableop_147_adam_predictionmask_kernel_vIdentity_147:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_147q
Identity_148IdentityRestoreV2:tensors:148"/device:CPU:0*
T0*
_output_shapes
:2
Identity_148�
AssignVariableOp_148AssignVariableOp/assignvariableop_148_adam_predictionmask_bias_vIdentity_148:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1489
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�
Identity_149Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_125^AssignVariableOp_126^AssignVariableOp_127^AssignVariableOp_128^AssignVariableOp_129^AssignVariableOp_13^AssignVariableOp_130^AssignVariableOp_131^AssignVariableOp_132^AssignVariableOp_133^AssignVariableOp_134^AssignVariableOp_135^AssignVariableOp_136^AssignVariableOp_137^AssignVariableOp_138^AssignVariableOp_139^AssignVariableOp_14^AssignVariableOp_140^AssignVariableOp_141^AssignVariableOp_142^AssignVariableOp_143^AssignVariableOp_144^AssignVariableOp_145^AssignVariableOp_146^AssignVariableOp_147^AssignVariableOp_148^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_149�
Identity_150IdentityIdentity_149:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_125^AssignVariableOp_126^AssignVariableOp_127^AssignVariableOp_128^AssignVariableOp_129^AssignVariableOp_13^AssignVariableOp_130^AssignVariableOp_131^AssignVariableOp_132^AssignVariableOp_133^AssignVariableOp_134^AssignVariableOp_135^AssignVariableOp_136^AssignVariableOp_137^AssignVariableOp_138^AssignVariableOp_139^AssignVariableOp_14^AssignVariableOp_140^AssignVariableOp_141^AssignVariableOp_142^AssignVariableOp_143^AssignVariableOp_144^AssignVariableOp_145^AssignVariableOp_146^AssignVariableOp_147^AssignVariableOp_148^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99*
T0*
_output_shapes
: 2
Identity_150"%
identity_150Identity_150:output:0*�
_input_shapes�
�: :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102,
AssignVariableOp_100AssignVariableOp_1002,
AssignVariableOp_101AssignVariableOp_1012,
AssignVariableOp_102AssignVariableOp_1022,
AssignVariableOp_103AssignVariableOp_1032,
AssignVariableOp_104AssignVariableOp_1042,
AssignVariableOp_105AssignVariableOp_1052,
AssignVariableOp_106AssignVariableOp_1062,
AssignVariableOp_107AssignVariableOp_1072,
AssignVariableOp_108AssignVariableOp_1082,
AssignVariableOp_109AssignVariableOp_1092*
AssignVariableOp_11AssignVariableOp_112,
AssignVariableOp_110AssignVariableOp_1102,
AssignVariableOp_111AssignVariableOp_1112,
AssignVariableOp_112AssignVariableOp_1122,
AssignVariableOp_113AssignVariableOp_1132,
AssignVariableOp_114AssignVariableOp_1142,
AssignVariableOp_115AssignVariableOp_1152,
AssignVariableOp_116AssignVariableOp_1162,
AssignVariableOp_117AssignVariableOp_1172,
AssignVariableOp_118AssignVariableOp_1182,
AssignVariableOp_119AssignVariableOp_1192*
AssignVariableOp_12AssignVariableOp_122,
AssignVariableOp_120AssignVariableOp_1202,
AssignVariableOp_121AssignVariableOp_1212,
AssignVariableOp_122AssignVariableOp_1222,
AssignVariableOp_123AssignVariableOp_1232,
AssignVariableOp_124AssignVariableOp_1242,
AssignVariableOp_125AssignVariableOp_1252,
AssignVariableOp_126AssignVariableOp_1262,
AssignVariableOp_127AssignVariableOp_1272,
AssignVariableOp_128AssignVariableOp_1282,
AssignVariableOp_129AssignVariableOp_1292*
AssignVariableOp_13AssignVariableOp_132,
AssignVariableOp_130AssignVariableOp_1302,
AssignVariableOp_131AssignVariableOp_1312,
AssignVariableOp_132AssignVariableOp_1322,
AssignVariableOp_133AssignVariableOp_1332,
AssignVariableOp_134AssignVariableOp_1342,
AssignVariableOp_135AssignVariableOp_1352,
AssignVariableOp_136AssignVariableOp_1362,
AssignVariableOp_137AssignVariableOp_1372,
AssignVariableOp_138AssignVariableOp_1382,
AssignVariableOp_139AssignVariableOp_1392*
AssignVariableOp_14AssignVariableOp_142,
AssignVariableOp_140AssignVariableOp_1402,
AssignVariableOp_141AssignVariableOp_1412,
AssignVariableOp_142AssignVariableOp_1422,
AssignVariableOp_143AssignVariableOp_1432,
AssignVariableOp_144AssignVariableOp_1442,
AssignVariableOp_145AssignVariableOp_1452,
AssignVariableOp_146AssignVariableOp_1462,
AssignVariableOp_147AssignVariableOp_1472,
AssignVariableOp_148AssignVariableOp_1482*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_86AssignVariableOp_862*
AssignVariableOp_87AssignVariableOp_872*
AssignVariableOp_88AssignVariableOp_882*
AssignVariableOp_89AssignVariableOp_892(
AssignVariableOp_9AssignVariableOp_92*
AssignVariableOp_90AssignVariableOp_902*
AssignVariableOp_91AssignVariableOp_912*
AssignVariableOp_92AssignVariableOp_922*
AssignVariableOp_93AssignVariableOp_932*
AssignVariableOp_94AssignVariableOp_942*
AssignVariableOp_95AssignVariableOp_952*
AssignVariableOp_96AssignVariableOp_962*
AssignVariableOp_97AssignVariableOp_972*
AssignVariableOp_98AssignVariableOp_982*
AssignVariableOp_99AssignVariableOp_99:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
m
N__inference_spatial_dropout2d_1_layer_call_and_return_conditional_losses_20665

inputs
identity�D
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dropout/Const�
dropout/MulMulinputsdropout/Const:output:0*
T0*J
_output_shapes8
6:4������������������������������������2
dropout/Mul�
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/1�
dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/2�
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0'dropout/random_uniform/shape/2:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
dropout/random_uniform/shape�
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*8
_output_shapes&
$:"������������������*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*8
_output_shapes&
$:"������������������2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*8
_output_shapes&
$:"������������������2
dropout/Cast�
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*J
_output_shapes8
6:4������������������������������������2
dropout/Mul_1�
IdentityIdentitydropout/Mul_1:z:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
}
(__inference_decodeCa_layer_call_fn_20773

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_decodeCa_layer_call_and_return_conditional_losses_187192
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
l
B__inference_concatD_layer_call_and_return_conditional_losses_18699

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis�
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*0
_output_shapes
:����������2
concatl
IdentityIdentityconcat:output:0*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*]
_input_shapesL
J:,����������������������������:����������:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs:XT
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
l
N__inference_spatial_dropout2d_1_layer_call_and_return_conditional_losses_18140

inputs

identity_1}
IdentityIdentityinputs*
T0*J
_output_shapes8
6:4������������������������������������2

Identity�

Identity_1IdentityIdentity:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity_1"!

identity_1Identity_1:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�

�
C__inference_decodeAb_layer_call_and_return_conditional_losses_20890

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@ *
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@ 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@@ 2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:���������@@ 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������@@ ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������@@ 
 
_user_specified_nameinputs
�

*__inference_transconvE_layer_call_fn_18199

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_transconvE_layer_call_and_return_conditional_losses_181892
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,����������������������������::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
A
%__inference_poolD_layer_call_fn_18155

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_poolD_layer_call_and_return_conditional_losses_181492
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
j
L__inference_spatial_dropout2d_layer_call_and_return_conditional_losses_20516

inputs

identity_1}
IdentityIdentityinputs*
T0*J
_output_shapes8
6:4������������������������������������2

Identity�

Identity_1IdentityIdentity:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity_1"!

identity_1Identity_1:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�

�
C__inference_convOutb_layer_call_and_return_conditional_losses_20943

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:�����������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:�����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�

�
C__inference_encodeAb_layer_call_and_return_conditional_losses_20419

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:�����������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:�����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
O
3__inference_spatial_dropout2d_1_layer_call_fn_20642

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_spatial_dropout2d_1_layer_call_and_return_conditional_losses_185932
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
n
B__inference_concatA_layer_call_and_return_conditional_losses_20906
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis�
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*1
_output_shapes
:����������� 2
concatm
IdentityIdentityconcat:output:0*
T0*1
_output_shapes
:����������� 2

Identity"
identityIdentity:output:0*]
_input_shapesL
J:+���������������������������:�����������:k g
A
_output_shapes/
-:+���������������������������
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:�����������
"
_user_specified_name
inputs/1
�
}
(__inference_encodeEa_layer_call_fn_20720

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_encodeEa_layer_call_and_return_conditional_losses_186442
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
}
(__inference_encodeAb_layer_call_fn_20428

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_encodeAb_layer_call_and_return_conditional_losses_183732
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:�����������::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�

�
C__inference_encodeAa_layer_call_and_return_conditional_losses_20399

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:�����������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:�����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�

�
C__inference_encodeCb_layer_call_and_return_conditional_losses_18522

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  @*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  @2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������  @2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:���������  @2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������  @::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������  @
 
_user_specified_nameinputs
�
M
1__inference_spatial_dropout2d_layer_call_fn_20564

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_spatial_dropout2d_layer_call_and_return_conditional_losses_184992
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������  @2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������  @:W S
/
_output_shapes
:���������  @
 
_user_specified_nameinputs
�
O
3__inference_spatial_dropout2d_1_layer_call_fn_20680

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_spatial_dropout2d_1_layer_call_and_return_conditional_losses_181402
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�

�
C__inference_decodeBa_layer_call_and_return_conditional_losses_20817

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:�@*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  @*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  @2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������  @2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:���������  @2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:���������  �::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:���������  �
 
_user_specified_nameinputs
�

*__inference_transconvB_layer_call_fn_18287

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+��������������������������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_transconvB_layer_call_and_return_conditional_losses_182772
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������@::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�

�
C__inference_decodeCb_layer_call_and_return_conditional_losses_18746

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:����������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
}
(__inference_decodeAb_layer_call_fn_20899

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_decodeAb_layer_call_and_return_conditional_losses_188962
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������@@ 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������@@ ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������@@ 
 
_user_specified_nameinputs
�

�
C__inference_encodeEa_layer_call_and_return_conditional_losses_20711

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:����������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
��
�.
 __inference__wrapped_model_17971
mrimagesA
=dunet_brats_decathlon_encodeaa_conv2d_readvariableop_resourceB
>dunet_brats_decathlon_encodeaa_biasadd_readvariableop_resourceA
=dunet_brats_decathlon_encodeab_conv2d_readvariableop_resourceB
>dunet_brats_decathlon_encodeab_biasadd_readvariableop_resourceA
=dunet_brats_decathlon_encodeba_conv2d_readvariableop_resourceB
>dunet_brats_decathlon_encodeba_biasadd_readvariableop_resourceA
=dunet_brats_decathlon_encodebb_conv2d_readvariableop_resourceB
>dunet_brats_decathlon_encodebb_biasadd_readvariableop_resourceA
=dunet_brats_decathlon_encodeca_conv2d_readvariableop_resourceB
>dunet_brats_decathlon_encodeca_biasadd_readvariableop_resourceA
=dunet_brats_decathlon_encodecb_conv2d_readvariableop_resourceB
>dunet_brats_decathlon_encodecb_biasadd_readvariableop_resourceA
=dunet_brats_decathlon_encodeda_conv2d_readvariableop_resourceB
>dunet_brats_decathlon_encodeda_biasadd_readvariableop_resourceA
=dunet_brats_decathlon_encodedb_conv2d_readvariableop_resourceB
>dunet_brats_decathlon_encodedb_biasadd_readvariableop_resourceA
=dunet_brats_decathlon_encodeea_conv2d_readvariableop_resourceB
>dunet_brats_decathlon_encodeea_biasadd_readvariableop_resourceA
=dunet_brats_decathlon_encodeeb_conv2d_readvariableop_resourceB
>dunet_brats_decathlon_encodeeb_biasadd_readvariableop_resourceM
Idunet_brats_decathlon_transconve_conv2d_transpose_readvariableop_resourceD
@dunet_brats_decathlon_transconve_biasadd_readvariableop_resourceA
=dunet_brats_decathlon_decodeca_conv2d_readvariableop_resourceB
>dunet_brats_decathlon_decodeca_biasadd_readvariableop_resourceA
=dunet_brats_decathlon_decodecb_conv2d_readvariableop_resourceB
>dunet_brats_decathlon_decodecb_biasadd_readvariableop_resourceM
Idunet_brats_decathlon_transconvc_conv2d_transpose_readvariableop_resourceD
@dunet_brats_decathlon_transconvc_biasadd_readvariableop_resourceA
=dunet_brats_decathlon_decodeba_conv2d_readvariableop_resourceB
>dunet_brats_decathlon_decodeba_biasadd_readvariableop_resourceA
=dunet_brats_decathlon_decodebb_conv2d_readvariableop_resourceB
>dunet_brats_decathlon_decodebb_biasadd_readvariableop_resourceM
Idunet_brats_decathlon_transconvb_conv2d_transpose_readvariableop_resourceD
@dunet_brats_decathlon_transconvb_biasadd_readvariableop_resourceA
=dunet_brats_decathlon_decodeaa_conv2d_readvariableop_resourceB
>dunet_brats_decathlon_decodeaa_biasadd_readvariableop_resourceA
=dunet_brats_decathlon_decodeab_conv2d_readvariableop_resourceB
>dunet_brats_decathlon_decodeab_biasadd_readvariableop_resourceM
Idunet_brats_decathlon_transconva_conv2d_transpose_readvariableop_resourceD
@dunet_brats_decathlon_transconva_biasadd_readvariableop_resourceA
=dunet_brats_decathlon_convouta_conv2d_readvariableop_resourceB
>dunet_brats_decathlon_convouta_biasadd_readvariableop_resourceA
=dunet_brats_decathlon_convoutb_conv2d_readvariableop_resourceB
>dunet_brats_decathlon_convoutb_biasadd_readvariableop_resourceG
Cdunet_brats_decathlon_predictionmask_conv2d_readvariableop_resourceH
Ddunet_brats_decathlon_predictionmask_biasadd_readvariableop_resource
identity��<2DUNet_Brats_Decathlon/PredictionMask/BiasAdd/ReadVariableOp�;2DUNet_Brats_Decathlon/PredictionMask/Conv2D/ReadVariableOp�62DUNet_Brats_Decathlon/convOuta/BiasAdd/ReadVariableOp�52DUNet_Brats_Decathlon/convOuta/Conv2D/ReadVariableOp�62DUNet_Brats_Decathlon/convOutb/BiasAdd/ReadVariableOp�52DUNet_Brats_Decathlon/convOutb/Conv2D/ReadVariableOp�62DUNet_Brats_Decathlon/decodeAa/BiasAdd/ReadVariableOp�52DUNet_Brats_Decathlon/decodeAa/Conv2D/ReadVariableOp�62DUNet_Brats_Decathlon/decodeAb/BiasAdd/ReadVariableOp�52DUNet_Brats_Decathlon/decodeAb/Conv2D/ReadVariableOp�62DUNet_Brats_Decathlon/decodeBa/BiasAdd/ReadVariableOp�52DUNet_Brats_Decathlon/decodeBa/Conv2D/ReadVariableOp�62DUNet_Brats_Decathlon/decodeBb/BiasAdd/ReadVariableOp�52DUNet_Brats_Decathlon/decodeBb/Conv2D/ReadVariableOp�62DUNet_Brats_Decathlon/decodeCa/BiasAdd/ReadVariableOp�52DUNet_Brats_Decathlon/decodeCa/Conv2D/ReadVariableOp�62DUNet_Brats_Decathlon/decodeCb/BiasAdd/ReadVariableOp�52DUNet_Brats_Decathlon/decodeCb/Conv2D/ReadVariableOp�62DUNet_Brats_Decathlon/encodeAa/BiasAdd/ReadVariableOp�52DUNet_Brats_Decathlon/encodeAa/Conv2D/ReadVariableOp�62DUNet_Brats_Decathlon/encodeAb/BiasAdd/ReadVariableOp�52DUNet_Brats_Decathlon/encodeAb/Conv2D/ReadVariableOp�62DUNet_Brats_Decathlon/encodeBa/BiasAdd/ReadVariableOp�52DUNet_Brats_Decathlon/encodeBa/Conv2D/ReadVariableOp�62DUNet_Brats_Decathlon/encodeBb/BiasAdd/ReadVariableOp�52DUNet_Brats_Decathlon/encodeBb/Conv2D/ReadVariableOp�62DUNet_Brats_Decathlon/encodeCa/BiasAdd/ReadVariableOp�52DUNet_Brats_Decathlon/encodeCa/Conv2D/ReadVariableOp�62DUNet_Brats_Decathlon/encodeCb/BiasAdd/ReadVariableOp�52DUNet_Brats_Decathlon/encodeCb/Conv2D/ReadVariableOp�62DUNet_Brats_Decathlon/encodeDa/BiasAdd/ReadVariableOp�52DUNet_Brats_Decathlon/encodeDa/Conv2D/ReadVariableOp�62DUNet_Brats_Decathlon/encodeDb/BiasAdd/ReadVariableOp�52DUNet_Brats_Decathlon/encodeDb/Conv2D/ReadVariableOp�62DUNet_Brats_Decathlon/encodeEa/BiasAdd/ReadVariableOp�52DUNet_Brats_Decathlon/encodeEa/Conv2D/ReadVariableOp�62DUNet_Brats_Decathlon/encodeEb/BiasAdd/ReadVariableOp�52DUNet_Brats_Decathlon/encodeEb/Conv2D/ReadVariableOp�82DUNet_Brats_Decathlon/transconvA/BiasAdd/ReadVariableOp�A2DUNet_Brats_Decathlon/transconvA/conv2d_transpose/ReadVariableOp�82DUNet_Brats_Decathlon/transconvB/BiasAdd/ReadVariableOp�A2DUNet_Brats_Decathlon/transconvB/conv2d_transpose/ReadVariableOp�82DUNet_Brats_Decathlon/transconvC/BiasAdd/ReadVariableOp�A2DUNet_Brats_Decathlon/transconvC/conv2d_transpose/ReadVariableOp�82DUNet_Brats_Decathlon/transconvE/BiasAdd/ReadVariableOp�A2DUNet_Brats_Decathlon/transconvE/conv2d_transpose/ReadVariableOp�
52DUNet_Brats_Decathlon/encodeAa/Conv2D/ReadVariableOpReadVariableOp=dunet_brats_decathlon_encodeaa_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype027
52DUNet_Brats_Decathlon/encodeAa/Conv2D/ReadVariableOp�
&2DUNet_Brats_Decathlon/encodeAa/Conv2DConv2Dmrimages=2DUNet_Brats_Decathlon/encodeAa/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
2(
&2DUNet_Brats_Decathlon/encodeAa/Conv2D�
62DUNet_Brats_Decathlon/encodeAa/BiasAdd/ReadVariableOpReadVariableOp>dunet_brats_decathlon_encodeaa_biasadd_readvariableop_resource*
_output_shapes
:*
dtype028
62DUNet_Brats_Decathlon/encodeAa/BiasAdd/ReadVariableOp�
'2DUNet_Brats_Decathlon/encodeAa/BiasAddBiasAdd/2DUNet_Brats_Decathlon/encodeAa/Conv2D:output:0>2DUNet_Brats_Decathlon/encodeAa/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������2)
'2DUNet_Brats_Decathlon/encodeAa/BiasAdd�
$2DUNet_Brats_Decathlon/encodeAa/ReluRelu02DUNet_Brats_Decathlon/encodeAa/BiasAdd:output:0*
T0*1
_output_shapes
:�����������2&
$2DUNet_Brats_Decathlon/encodeAa/Relu�
52DUNet_Brats_Decathlon/encodeAb/Conv2D/ReadVariableOpReadVariableOp=dunet_brats_decathlon_encodeab_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype027
52DUNet_Brats_Decathlon/encodeAb/Conv2D/ReadVariableOp�
&2DUNet_Brats_Decathlon/encodeAb/Conv2DConv2D22DUNet_Brats_Decathlon/encodeAa/Relu:activations:0=2DUNet_Brats_Decathlon/encodeAb/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
2(
&2DUNet_Brats_Decathlon/encodeAb/Conv2D�
62DUNet_Brats_Decathlon/encodeAb/BiasAdd/ReadVariableOpReadVariableOp>dunet_brats_decathlon_encodeab_biasadd_readvariableop_resource*
_output_shapes
:*
dtype028
62DUNet_Brats_Decathlon/encodeAb/BiasAdd/ReadVariableOp�
'2DUNet_Brats_Decathlon/encodeAb/BiasAddBiasAdd/2DUNet_Brats_Decathlon/encodeAb/Conv2D:output:0>2DUNet_Brats_Decathlon/encodeAb/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������2)
'2DUNet_Brats_Decathlon/encodeAb/BiasAdd�
$2DUNet_Brats_Decathlon/encodeAb/ReluRelu02DUNet_Brats_Decathlon/encodeAb/BiasAdd:output:0*
T0*1
_output_shapes
:�����������2&
$2DUNet_Brats_Decathlon/encodeAb/Relu�
$2DUNet_Brats_Decathlon/poolA/MaxPoolMaxPool22DUNet_Brats_Decathlon/encodeAb/Relu:activations:0*/
_output_shapes
:���������@@*
ksize
*
paddingVALID*
strides
2&
$2DUNet_Brats_Decathlon/poolA/MaxPool�
52DUNet_Brats_Decathlon/encodeBa/Conv2D/ReadVariableOpReadVariableOp=dunet_brats_decathlon_encodeba_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype027
52DUNet_Brats_Decathlon/encodeBa/Conv2D/ReadVariableOp�
&2DUNet_Brats_Decathlon/encodeBa/Conv2DConv2D-2DUNet_Brats_Decathlon/poolA/MaxPool:output:0=2DUNet_Brats_Decathlon/encodeBa/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@ *
paddingSAME*
strides
2(
&2DUNet_Brats_Decathlon/encodeBa/Conv2D�
62DUNet_Brats_Decathlon/encodeBa/BiasAdd/ReadVariableOpReadVariableOp>dunet_brats_decathlon_encodeba_biasadd_readvariableop_resource*
_output_shapes
: *
dtype028
62DUNet_Brats_Decathlon/encodeBa/BiasAdd/ReadVariableOp�
'2DUNet_Brats_Decathlon/encodeBa/BiasAddBiasAdd/2DUNet_Brats_Decathlon/encodeBa/Conv2D:output:0>2DUNet_Brats_Decathlon/encodeBa/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@ 2)
'2DUNet_Brats_Decathlon/encodeBa/BiasAdd�
$2DUNet_Brats_Decathlon/encodeBa/ReluRelu02DUNet_Brats_Decathlon/encodeBa/BiasAdd:output:0*
T0*/
_output_shapes
:���������@@ 2&
$2DUNet_Brats_Decathlon/encodeBa/Relu�
52DUNet_Brats_Decathlon/encodeBb/Conv2D/ReadVariableOpReadVariableOp=dunet_brats_decathlon_encodebb_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype027
52DUNet_Brats_Decathlon/encodeBb/Conv2D/ReadVariableOp�
&2DUNet_Brats_Decathlon/encodeBb/Conv2DConv2D22DUNet_Brats_Decathlon/encodeBa/Relu:activations:0=2DUNet_Brats_Decathlon/encodeBb/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@ *
paddingSAME*
strides
2(
&2DUNet_Brats_Decathlon/encodeBb/Conv2D�
62DUNet_Brats_Decathlon/encodeBb/BiasAdd/ReadVariableOpReadVariableOp>dunet_brats_decathlon_encodebb_biasadd_readvariableop_resource*
_output_shapes
: *
dtype028
62DUNet_Brats_Decathlon/encodeBb/BiasAdd/ReadVariableOp�
'2DUNet_Brats_Decathlon/encodeBb/BiasAddBiasAdd/2DUNet_Brats_Decathlon/encodeBb/Conv2D:output:0>2DUNet_Brats_Decathlon/encodeBb/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@ 2)
'2DUNet_Brats_Decathlon/encodeBb/BiasAdd�
$2DUNet_Brats_Decathlon/encodeBb/ReluRelu02DUNet_Brats_Decathlon/encodeBb/BiasAdd:output:0*
T0*/
_output_shapes
:���������@@ 2&
$2DUNet_Brats_Decathlon/encodeBb/Relu�
$2DUNet_Brats_Decathlon/poolB/MaxPoolMaxPool22DUNet_Brats_Decathlon/encodeBb/Relu:activations:0*/
_output_shapes
:���������   *
ksize
*
paddingVALID*
strides
2&
$2DUNet_Brats_Decathlon/poolB/MaxPool�
52DUNet_Brats_Decathlon/encodeCa/Conv2D/ReadVariableOpReadVariableOp=dunet_brats_decathlon_encodeca_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype027
52DUNet_Brats_Decathlon/encodeCa/Conv2D/ReadVariableOp�
&2DUNet_Brats_Decathlon/encodeCa/Conv2DConv2D-2DUNet_Brats_Decathlon/poolB/MaxPool:output:0=2DUNet_Brats_Decathlon/encodeCa/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  @*
paddingSAME*
strides
2(
&2DUNet_Brats_Decathlon/encodeCa/Conv2D�
62DUNet_Brats_Decathlon/encodeCa/BiasAdd/ReadVariableOpReadVariableOp>dunet_brats_decathlon_encodeca_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype028
62DUNet_Brats_Decathlon/encodeCa/BiasAdd/ReadVariableOp�
'2DUNet_Brats_Decathlon/encodeCa/BiasAddBiasAdd/2DUNet_Brats_Decathlon/encodeCa/Conv2D:output:0>2DUNet_Brats_Decathlon/encodeCa/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  @2)
'2DUNet_Brats_Decathlon/encodeCa/BiasAdd�
$2DUNet_Brats_Decathlon/encodeCa/ReluRelu02DUNet_Brats_Decathlon/encodeCa/BiasAdd:output:0*
T0*/
_output_shapes
:���������  @2&
$2DUNet_Brats_Decathlon/encodeCa/Relu�
12DUNet_Brats_Decathlon/spatial_dropout2d/IdentityIdentity22DUNet_Brats_Decathlon/encodeCa/Relu:activations:0*
T0*/
_output_shapes
:���������  @23
12DUNet_Brats_Decathlon/spatial_dropout2d/Identity�
52DUNet_Brats_Decathlon/encodeCb/Conv2D/ReadVariableOpReadVariableOp=dunet_brats_decathlon_encodecb_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype027
52DUNet_Brats_Decathlon/encodeCb/Conv2D/ReadVariableOp�
&2DUNet_Brats_Decathlon/encodeCb/Conv2DConv2D:2DUNet_Brats_Decathlon/spatial_dropout2d/Identity:output:0=2DUNet_Brats_Decathlon/encodeCb/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  @*
paddingSAME*
strides
2(
&2DUNet_Brats_Decathlon/encodeCb/Conv2D�
62DUNet_Brats_Decathlon/encodeCb/BiasAdd/ReadVariableOpReadVariableOp>dunet_brats_decathlon_encodecb_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype028
62DUNet_Brats_Decathlon/encodeCb/BiasAdd/ReadVariableOp�
'2DUNet_Brats_Decathlon/encodeCb/BiasAddBiasAdd/2DUNet_Brats_Decathlon/encodeCb/Conv2D:output:0>2DUNet_Brats_Decathlon/encodeCb/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  @2)
'2DUNet_Brats_Decathlon/encodeCb/BiasAdd�
$2DUNet_Brats_Decathlon/encodeCb/ReluRelu02DUNet_Brats_Decathlon/encodeCb/BiasAdd:output:0*
T0*/
_output_shapes
:���������  @2&
$2DUNet_Brats_Decathlon/encodeCb/Relu�
$2DUNet_Brats_Decathlon/poolC/MaxPoolMaxPool22DUNet_Brats_Decathlon/encodeCb/Relu:activations:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
2&
$2DUNet_Brats_Decathlon/poolC/MaxPool�
52DUNet_Brats_Decathlon/encodeDa/Conv2D/ReadVariableOpReadVariableOp=dunet_brats_decathlon_encodeda_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype027
52DUNet_Brats_Decathlon/encodeDa/Conv2D/ReadVariableOp�
&2DUNet_Brats_Decathlon/encodeDa/Conv2DConv2D-2DUNet_Brats_Decathlon/poolC/MaxPool:output:0=2DUNet_Brats_Decathlon/encodeDa/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2(
&2DUNet_Brats_Decathlon/encodeDa/Conv2D�
62DUNet_Brats_Decathlon/encodeDa/BiasAdd/ReadVariableOpReadVariableOp>dunet_brats_decathlon_encodeda_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype028
62DUNet_Brats_Decathlon/encodeDa/BiasAdd/ReadVariableOp�
'2DUNet_Brats_Decathlon/encodeDa/BiasAddBiasAdd/2DUNet_Brats_Decathlon/encodeDa/Conv2D:output:0>2DUNet_Brats_Decathlon/encodeDa/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2)
'2DUNet_Brats_Decathlon/encodeDa/BiasAdd�
$2DUNet_Brats_Decathlon/encodeDa/ReluRelu02DUNet_Brats_Decathlon/encodeDa/BiasAdd:output:0*
T0*0
_output_shapes
:����������2&
$2DUNet_Brats_Decathlon/encodeDa/Relu�
32DUNet_Brats_Decathlon/spatial_dropout2d_1/IdentityIdentity22DUNet_Brats_Decathlon/encodeDa/Relu:activations:0*
T0*0
_output_shapes
:����������25
32DUNet_Brats_Decathlon/spatial_dropout2d_1/Identity�
52DUNet_Brats_Decathlon/encodeDb/Conv2D/ReadVariableOpReadVariableOp=dunet_brats_decathlon_encodedb_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype027
52DUNet_Brats_Decathlon/encodeDb/Conv2D/ReadVariableOp�
&2DUNet_Brats_Decathlon/encodeDb/Conv2DConv2D<2DUNet_Brats_Decathlon/spatial_dropout2d_1/Identity:output:0=2DUNet_Brats_Decathlon/encodeDb/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2(
&2DUNet_Brats_Decathlon/encodeDb/Conv2D�
62DUNet_Brats_Decathlon/encodeDb/BiasAdd/ReadVariableOpReadVariableOp>dunet_brats_decathlon_encodedb_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype028
62DUNet_Brats_Decathlon/encodeDb/BiasAdd/ReadVariableOp�
'2DUNet_Brats_Decathlon/encodeDb/BiasAddBiasAdd/2DUNet_Brats_Decathlon/encodeDb/Conv2D:output:0>2DUNet_Brats_Decathlon/encodeDb/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2)
'2DUNet_Brats_Decathlon/encodeDb/BiasAdd�
$2DUNet_Brats_Decathlon/encodeDb/ReluRelu02DUNet_Brats_Decathlon/encodeDb/BiasAdd:output:0*
T0*0
_output_shapes
:����������2&
$2DUNet_Brats_Decathlon/encodeDb/Relu�
$2DUNet_Brats_Decathlon/poolD/MaxPoolMaxPool22DUNet_Brats_Decathlon/encodeDb/Relu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2&
$2DUNet_Brats_Decathlon/poolD/MaxPool�
52DUNet_Brats_Decathlon/encodeEa/Conv2D/ReadVariableOpReadVariableOp=dunet_brats_decathlon_encodeea_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype027
52DUNet_Brats_Decathlon/encodeEa/Conv2D/ReadVariableOp�
&2DUNet_Brats_Decathlon/encodeEa/Conv2DConv2D-2DUNet_Brats_Decathlon/poolD/MaxPool:output:0=2DUNet_Brats_Decathlon/encodeEa/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2(
&2DUNet_Brats_Decathlon/encodeEa/Conv2D�
62DUNet_Brats_Decathlon/encodeEa/BiasAdd/ReadVariableOpReadVariableOp>dunet_brats_decathlon_encodeea_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype028
62DUNet_Brats_Decathlon/encodeEa/BiasAdd/ReadVariableOp�
'2DUNet_Brats_Decathlon/encodeEa/BiasAddBiasAdd/2DUNet_Brats_Decathlon/encodeEa/Conv2D:output:0>2DUNet_Brats_Decathlon/encodeEa/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2)
'2DUNet_Brats_Decathlon/encodeEa/BiasAdd�
$2DUNet_Brats_Decathlon/encodeEa/ReluRelu02DUNet_Brats_Decathlon/encodeEa/BiasAdd:output:0*
T0*0
_output_shapes
:����������2&
$2DUNet_Brats_Decathlon/encodeEa/Relu�
52DUNet_Brats_Decathlon/encodeEb/Conv2D/ReadVariableOpReadVariableOp=dunet_brats_decathlon_encodeeb_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype027
52DUNet_Brats_Decathlon/encodeEb/Conv2D/ReadVariableOp�
&2DUNet_Brats_Decathlon/encodeEb/Conv2DConv2D22DUNet_Brats_Decathlon/encodeEa/Relu:activations:0=2DUNet_Brats_Decathlon/encodeEb/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2(
&2DUNet_Brats_Decathlon/encodeEb/Conv2D�
62DUNet_Brats_Decathlon/encodeEb/BiasAdd/ReadVariableOpReadVariableOp>dunet_brats_decathlon_encodeeb_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype028
62DUNet_Brats_Decathlon/encodeEb/BiasAdd/ReadVariableOp�
'2DUNet_Brats_Decathlon/encodeEb/BiasAddBiasAdd/2DUNet_Brats_Decathlon/encodeEb/Conv2D:output:0>2DUNet_Brats_Decathlon/encodeEb/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2)
'2DUNet_Brats_Decathlon/encodeEb/BiasAdd�
$2DUNet_Brats_Decathlon/encodeEb/ReluRelu02DUNet_Brats_Decathlon/encodeEb/BiasAdd:output:0*
T0*0
_output_shapes
:����������2&
$2DUNet_Brats_Decathlon/encodeEb/Relu�
'2DUNet_Brats_Decathlon/transconvE/ShapeShape22DUNet_Brats_Decathlon/encodeEb/Relu:activations:0*
T0*
_output_shapes
:2)
'2DUNet_Brats_Decathlon/transconvE/Shape�
52DUNet_Brats_Decathlon/transconvE/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
52DUNet_Brats_Decathlon/transconvE/strided_slice/stack�
72DUNet_Brats_Decathlon/transconvE/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
72DUNet_Brats_Decathlon/transconvE/strided_slice/stack_1�
72DUNet_Brats_Decathlon/transconvE/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
72DUNet_Brats_Decathlon/transconvE/strided_slice/stack_2�
/2DUNet_Brats_Decathlon/transconvE/strided_sliceStridedSlice02DUNet_Brats_Decathlon/transconvE/Shape:output:0>2DUNet_Brats_Decathlon/transconvE/strided_slice/stack:output:0@2DUNet_Brats_Decathlon/transconvE/strided_slice/stack_1:output:0@2DUNet_Brats_Decathlon/transconvE/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/2DUNet_Brats_Decathlon/transconvE/strided_slice�
)2DUNet_Brats_Decathlon/transconvE/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2+
)2DUNet_Brats_Decathlon/transconvE/stack/1�
)2DUNet_Brats_Decathlon/transconvE/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2+
)2DUNet_Brats_Decathlon/transconvE/stack/2�
)2DUNet_Brats_Decathlon/transconvE/stack/3Const*
_output_shapes
: *
dtype0*
value
B :�2+
)2DUNet_Brats_Decathlon/transconvE/stack/3�
'2DUNet_Brats_Decathlon/transconvE/stackPack82DUNet_Brats_Decathlon/transconvE/strided_slice:output:022DUNet_Brats_Decathlon/transconvE/stack/1:output:022DUNet_Brats_Decathlon/transconvE/stack/2:output:022DUNet_Brats_Decathlon/transconvE/stack/3:output:0*
N*
T0*
_output_shapes
:2)
'2DUNet_Brats_Decathlon/transconvE/stack�
72DUNet_Brats_Decathlon/transconvE/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 29
72DUNet_Brats_Decathlon/transconvE/strided_slice_1/stack�
92DUNet_Brats_Decathlon/transconvE/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
92DUNet_Brats_Decathlon/transconvE/strided_slice_1/stack_1�
92DUNet_Brats_Decathlon/transconvE/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
92DUNet_Brats_Decathlon/transconvE/strided_slice_1/stack_2�
12DUNet_Brats_Decathlon/transconvE/strided_slice_1StridedSlice02DUNet_Brats_Decathlon/transconvE/stack:output:0@2DUNet_Brats_Decathlon/transconvE/strided_slice_1/stack:output:0B2DUNet_Brats_Decathlon/transconvE/strided_slice_1/stack_1:output:0B2DUNet_Brats_Decathlon/transconvE/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask23
12DUNet_Brats_Decathlon/transconvE/strided_slice_1�
A2DUNet_Brats_Decathlon/transconvE/conv2d_transpose/ReadVariableOpReadVariableOpIdunet_brats_decathlon_transconve_conv2d_transpose_readvariableop_resource*(
_output_shapes
:��*
dtype02C
A2DUNet_Brats_Decathlon/transconvE/conv2d_transpose/ReadVariableOp�
22DUNet_Brats_Decathlon/transconvE/conv2d_transposeConv2DBackpropInput02DUNet_Brats_Decathlon/transconvE/stack:output:0I2DUNet_Brats_Decathlon/transconvE/conv2d_transpose/ReadVariableOp:value:022DUNet_Brats_Decathlon/encodeEb/Relu:activations:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
24
22DUNet_Brats_Decathlon/transconvE/conv2d_transpose�
82DUNet_Brats_Decathlon/transconvE/BiasAdd/ReadVariableOpReadVariableOp@dunet_brats_decathlon_transconve_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02:
82DUNet_Brats_Decathlon/transconvE/BiasAdd/ReadVariableOp�
)2DUNet_Brats_Decathlon/transconvE/BiasAddBiasAdd;2DUNet_Brats_Decathlon/transconvE/conv2d_transpose:output:0@2DUNet_Brats_Decathlon/transconvE/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2+
)2DUNet_Brats_Decathlon/transconvE/BiasAdd�
*2DUNet_Brats_Decathlon/concatD/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2,
*2DUNet_Brats_Decathlon/concatD/concat/axis�
%2DUNet_Brats_Decathlon/concatD/concatConcatV222DUNet_Brats_Decathlon/transconvE/BiasAdd:output:022DUNet_Brats_Decathlon/encodeDb/Relu:activations:032DUNet_Brats_Decathlon/concatD/concat/axis:output:0*
N*
T0*0
_output_shapes
:����������2'
%2DUNet_Brats_Decathlon/concatD/concat�
52DUNet_Brats_Decathlon/decodeCa/Conv2D/ReadVariableOpReadVariableOp=dunet_brats_decathlon_decodeca_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype027
52DUNet_Brats_Decathlon/decodeCa/Conv2D/ReadVariableOp�
&2DUNet_Brats_Decathlon/decodeCa/Conv2DConv2D.2DUNet_Brats_Decathlon/concatD/concat:output:0=2DUNet_Brats_Decathlon/decodeCa/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2(
&2DUNet_Brats_Decathlon/decodeCa/Conv2D�
62DUNet_Brats_Decathlon/decodeCa/BiasAdd/ReadVariableOpReadVariableOp>dunet_brats_decathlon_decodeca_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype028
62DUNet_Brats_Decathlon/decodeCa/BiasAdd/ReadVariableOp�
'2DUNet_Brats_Decathlon/decodeCa/BiasAddBiasAdd/2DUNet_Brats_Decathlon/decodeCa/Conv2D:output:0>2DUNet_Brats_Decathlon/decodeCa/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2)
'2DUNet_Brats_Decathlon/decodeCa/BiasAdd�
$2DUNet_Brats_Decathlon/decodeCa/ReluRelu02DUNet_Brats_Decathlon/decodeCa/BiasAdd:output:0*
T0*0
_output_shapes
:����������2&
$2DUNet_Brats_Decathlon/decodeCa/Relu�
52DUNet_Brats_Decathlon/decodeCb/Conv2D/ReadVariableOpReadVariableOp=dunet_brats_decathlon_decodecb_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype027
52DUNet_Brats_Decathlon/decodeCb/Conv2D/ReadVariableOp�
&2DUNet_Brats_Decathlon/decodeCb/Conv2DConv2D22DUNet_Brats_Decathlon/decodeCa/Relu:activations:0=2DUNet_Brats_Decathlon/decodeCb/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2(
&2DUNet_Brats_Decathlon/decodeCb/Conv2D�
62DUNet_Brats_Decathlon/decodeCb/BiasAdd/ReadVariableOpReadVariableOp>dunet_brats_decathlon_decodecb_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype028
62DUNet_Brats_Decathlon/decodeCb/BiasAdd/ReadVariableOp�
'2DUNet_Brats_Decathlon/decodeCb/BiasAddBiasAdd/2DUNet_Brats_Decathlon/decodeCb/Conv2D:output:0>2DUNet_Brats_Decathlon/decodeCb/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2)
'2DUNet_Brats_Decathlon/decodeCb/BiasAdd�
$2DUNet_Brats_Decathlon/decodeCb/ReluRelu02DUNet_Brats_Decathlon/decodeCb/BiasAdd:output:0*
T0*0
_output_shapes
:����������2&
$2DUNet_Brats_Decathlon/decodeCb/Relu�
'2DUNet_Brats_Decathlon/transconvC/ShapeShape22DUNet_Brats_Decathlon/decodeCb/Relu:activations:0*
T0*
_output_shapes
:2)
'2DUNet_Brats_Decathlon/transconvC/Shape�
52DUNet_Brats_Decathlon/transconvC/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
52DUNet_Brats_Decathlon/transconvC/strided_slice/stack�
72DUNet_Brats_Decathlon/transconvC/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
72DUNet_Brats_Decathlon/transconvC/strided_slice/stack_1�
72DUNet_Brats_Decathlon/transconvC/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
72DUNet_Brats_Decathlon/transconvC/strided_slice/stack_2�
/2DUNet_Brats_Decathlon/transconvC/strided_sliceStridedSlice02DUNet_Brats_Decathlon/transconvC/Shape:output:0>2DUNet_Brats_Decathlon/transconvC/strided_slice/stack:output:0@2DUNet_Brats_Decathlon/transconvC/strided_slice/stack_1:output:0@2DUNet_Brats_Decathlon/transconvC/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/2DUNet_Brats_Decathlon/transconvC/strided_slice�
)2DUNet_Brats_Decathlon/transconvC/stack/1Const*
_output_shapes
: *
dtype0*
value	B : 2+
)2DUNet_Brats_Decathlon/transconvC/stack/1�
)2DUNet_Brats_Decathlon/transconvC/stack/2Const*
_output_shapes
: *
dtype0*
value	B : 2+
)2DUNet_Brats_Decathlon/transconvC/stack/2�
)2DUNet_Brats_Decathlon/transconvC/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2+
)2DUNet_Brats_Decathlon/transconvC/stack/3�
'2DUNet_Brats_Decathlon/transconvC/stackPack82DUNet_Brats_Decathlon/transconvC/strided_slice:output:022DUNet_Brats_Decathlon/transconvC/stack/1:output:022DUNet_Brats_Decathlon/transconvC/stack/2:output:022DUNet_Brats_Decathlon/transconvC/stack/3:output:0*
N*
T0*
_output_shapes
:2)
'2DUNet_Brats_Decathlon/transconvC/stack�
72DUNet_Brats_Decathlon/transconvC/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 29
72DUNet_Brats_Decathlon/transconvC/strided_slice_1/stack�
92DUNet_Brats_Decathlon/transconvC/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
92DUNet_Brats_Decathlon/transconvC/strided_slice_1/stack_1�
92DUNet_Brats_Decathlon/transconvC/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
92DUNet_Brats_Decathlon/transconvC/strided_slice_1/stack_2�
12DUNet_Brats_Decathlon/transconvC/strided_slice_1StridedSlice02DUNet_Brats_Decathlon/transconvC/stack:output:0@2DUNet_Brats_Decathlon/transconvC/strided_slice_1/stack:output:0B2DUNet_Brats_Decathlon/transconvC/strided_slice_1/stack_1:output:0B2DUNet_Brats_Decathlon/transconvC/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask23
12DUNet_Brats_Decathlon/transconvC/strided_slice_1�
A2DUNet_Brats_Decathlon/transconvC/conv2d_transpose/ReadVariableOpReadVariableOpIdunet_brats_decathlon_transconvc_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@�*
dtype02C
A2DUNet_Brats_Decathlon/transconvC/conv2d_transpose/ReadVariableOp�
22DUNet_Brats_Decathlon/transconvC/conv2d_transposeConv2DBackpropInput02DUNet_Brats_Decathlon/transconvC/stack:output:0I2DUNet_Brats_Decathlon/transconvC/conv2d_transpose/ReadVariableOp:value:022DUNet_Brats_Decathlon/decodeCb/Relu:activations:0*
T0*/
_output_shapes
:���������  @*
paddingSAME*
strides
24
22DUNet_Brats_Decathlon/transconvC/conv2d_transpose�
82DUNet_Brats_Decathlon/transconvC/BiasAdd/ReadVariableOpReadVariableOp@dunet_brats_decathlon_transconvc_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02:
82DUNet_Brats_Decathlon/transconvC/BiasAdd/ReadVariableOp�
)2DUNet_Brats_Decathlon/transconvC/BiasAddBiasAdd;2DUNet_Brats_Decathlon/transconvC/conv2d_transpose:output:0@2DUNet_Brats_Decathlon/transconvC/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  @2+
)2DUNet_Brats_Decathlon/transconvC/BiasAdd�
*2DUNet_Brats_Decathlon/concatC/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2,
*2DUNet_Brats_Decathlon/concatC/concat/axis�
%2DUNet_Brats_Decathlon/concatC/concatConcatV222DUNet_Brats_Decathlon/transconvC/BiasAdd:output:022DUNet_Brats_Decathlon/encodeCb/Relu:activations:032DUNet_Brats_Decathlon/concatC/concat/axis:output:0*
N*
T0*0
_output_shapes
:���������  �2'
%2DUNet_Brats_Decathlon/concatC/concat�
52DUNet_Brats_Decathlon/decodeBa/Conv2D/ReadVariableOpReadVariableOp=dunet_brats_decathlon_decodeba_conv2d_readvariableop_resource*'
_output_shapes
:�@*
dtype027
52DUNet_Brats_Decathlon/decodeBa/Conv2D/ReadVariableOp�
&2DUNet_Brats_Decathlon/decodeBa/Conv2DConv2D.2DUNet_Brats_Decathlon/concatC/concat:output:0=2DUNet_Brats_Decathlon/decodeBa/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  @*
paddingSAME*
strides
2(
&2DUNet_Brats_Decathlon/decodeBa/Conv2D�
62DUNet_Brats_Decathlon/decodeBa/BiasAdd/ReadVariableOpReadVariableOp>dunet_brats_decathlon_decodeba_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype028
62DUNet_Brats_Decathlon/decodeBa/BiasAdd/ReadVariableOp�
'2DUNet_Brats_Decathlon/decodeBa/BiasAddBiasAdd/2DUNet_Brats_Decathlon/decodeBa/Conv2D:output:0>2DUNet_Brats_Decathlon/decodeBa/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  @2)
'2DUNet_Brats_Decathlon/decodeBa/BiasAdd�
$2DUNet_Brats_Decathlon/decodeBa/ReluRelu02DUNet_Brats_Decathlon/decodeBa/BiasAdd:output:0*
T0*/
_output_shapes
:���������  @2&
$2DUNet_Brats_Decathlon/decodeBa/Relu�
52DUNet_Brats_Decathlon/decodeBb/Conv2D/ReadVariableOpReadVariableOp=dunet_brats_decathlon_decodebb_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype027
52DUNet_Brats_Decathlon/decodeBb/Conv2D/ReadVariableOp�
&2DUNet_Brats_Decathlon/decodeBb/Conv2DConv2D22DUNet_Brats_Decathlon/decodeBa/Relu:activations:0=2DUNet_Brats_Decathlon/decodeBb/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  @*
paddingSAME*
strides
2(
&2DUNet_Brats_Decathlon/decodeBb/Conv2D�
62DUNet_Brats_Decathlon/decodeBb/BiasAdd/ReadVariableOpReadVariableOp>dunet_brats_decathlon_decodebb_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype028
62DUNet_Brats_Decathlon/decodeBb/BiasAdd/ReadVariableOp�
'2DUNet_Brats_Decathlon/decodeBb/BiasAddBiasAdd/2DUNet_Brats_Decathlon/decodeBb/Conv2D:output:0>2DUNet_Brats_Decathlon/decodeBb/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  @2)
'2DUNet_Brats_Decathlon/decodeBb/BiasAdd�
$2DUNet_Brats_Decathlon/decodeBb/ReluRelu02DUNet_Brats_Decathlon/decodeBb/BiasAdd:output:0*
T0*/
_output_shapes
:���������  @2&
$2DUNet_Brats_Decathlon/decodeBb/Relu�
'2DUNet_Brats_Decathlon/transconvB/ShapeShape22DUNet_Brats_Decathlon/decodeBb/Relu:activations:0*
T0*
_output_shapes
:2)
'2DUNet_Brats_Decathlon/transconvB/Shape�
52DUNet_Brats_Decathlon/transconvB/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
52DUNet_Brats_Decathlon/transconvB/strided_slice/stack�
72DUNet_Brats_Decathlon/transconvB/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
72DUNet_Brats_Decathlon/transconvB/strided_slice/stack_1�
72DUNet_Brats_Decathlon/transconvB/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
72DUNet_Brats_Decathlon/transconvB/strided_slice/stack_2�
/2DUNet_Brats_Decathlon/transconvB/strided_sliceStridedSlice02DUNet_Brats_Decathlon/transconvB/Shape:output:0>2DUNet_Brats_Decathlon/transconvB/strided_slice/stack:output:0@2DUNet_Brats_Decathlon/transconvB/strided_slice/stack_1:output:0@2DUNet_Brats_Decathlon/transconvB/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/2DUNet_Brats_Decathlon/transconvB/strided_slice�
)2DUNet_Brats_Decathlon/transconvB/stack/1Const*
_output_shapes
: *
dtype0*
value	B :@2+
)2DUNet_Brats_Decathlon/transconvB/stack/1�
)2DUNet_Brats_Decathlon/transconvB/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@2+
)2DUNet_Brats_Decathlon/transconvB/stack/2�
)2DUNet_Brats_Decathlon/transconvB/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2+
)2DUNet_Brats_Decathlon/transconvB/stack/3�
'2DUNet_Brats_Decathlon/transconvB/stackPack82DUNet_Brats_Decathlon/transconvB/strided_slice:output:022DUNet_Brats_Decathlon/transconvB/stack/1:output:022DUNet_Brats_Decathlon/transconvB/stack/2:output:022DUNet_Brats_Decathlon/transconvB/stack/3:output:0*
N*
T0*
_output_shapes
:2)
'2DUNet_Brats_Decathlon/transconvB/stack�
72DUNet_Brats_Decathlon/transconvB/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 29
72DUNet_Brats_Decathlon/transconvB/strided_slice_1/stack�
92DUNet_Brats_Decathlon/transconvB/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
92DUNet_Brats_Decathlon/transconvB/strided_slice_1/stack_1�
92DUNet_Brats_Decathlon/transconvB/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
92DUNet_Brats_Decathlon/transconvB/strided_slice_1/stack_2�
12DUNet_Brats_Decathlon/transconvB/strided_slice_1StridedSlice02DUNet_Brats_Decathlon/transconvB/stack:output:0@2DUNet_Brats_Decathlon/transconvB/strided_slice_1/stack:output:0B2DUNet_Brats_Decathlon/transconvB/strided_slice_1/stack_1:output:0B2DUNet_Brats_Decathlon/transconvB/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask23
12DUNet_Brats_Decathlon/transconvB/strided_slice_1�
A2DUNet_Brats_Decathlon/transconvB/conv2d_transpose/ReadVariableOpReadVariableOpIdunet_brats_decathlon_transconvb_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype02C
A2DUNet_Brats_Decathlon/transconvB/conv2d_transpose/ReadVariableOp�
22DUNet_Brats_Decathlon/transconvB/conv2d_transposeConv2DBackpropInput02DUNet_Brats_Decathlon/transconvB/stack:output:0I2DUNet_Brats_Decathlon/transconvB/conv2d_transpose/ReadVariableOp:value:022DUNet_Brats_Decathlon/decodeBb/Relu:activations:0*
T0*/
_output_shapes
:���������@@ *
paddingSAME*
strides
24
22DUNet_Brats_Decathlon/transconvB/conv2d_transpose�
82DUNet_Brats_Decathlon/transconvB/BiasAdd/ReadVariableOpReadVariableOp@dunet_brats_decathlon_transconvb_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02:
82DUNet_Brats_Decathlon/transconvB/BiasAdd/ReadVariableOp�
)2DUNet_Brats_Decathlon/transconvB/BiasAddBiasAdd;2DUNet_Brats_Decathlon/transconvB/conv2d_transpose:output:0@2DUNet_Brats_Decathlon/transconvB/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@ 2+
)2DUNet_Brats_Decathlon/transconvB/BiasAdd�
*2DUNet_Brats_Decathlon/concatB/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2,
*2DUNet_Brats_Decathlon/concatB/concat/axis�
%2DUNet_Brats_Decathlon/concatB/concatConcatV222DUNet_Brats_Decathlon/transconvB/BiasAdd:output:022DUNet_Brats_Decathlon/encodeBb/Relu:activations:032DUNet_Brats_Decathlon/concatB/concat/axis:output:0*
N*
T0*/
_output_shapes
:���������@@@2'
%2DUNet_Brats_Decathlon/concatB/concat�
52DUNet_Brats_Decathlon/decodeAa/Conv2D/ReadVariableOpReadVariableOp=dunet_brats_decathlon_decodeaa_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype027
52DUNet_Brats_Decathlon/decodeAa/Conv2D/ReadVariableOp�
&2DUNet_Brats_Decathlon/decodeAa/Conv2DConv2D.2DUNet_Brats_Decathlon/concatB/concat:output:0=2DUNet_Brats_Decathlon/decodeAa/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@ *
paddingSAME*
strides
2(
&2DUNet_Brats_Decathlon/decodeAa/Conv2D�
62DUNet_Brats_Decathlon/decodeAa/BiasAdd/ReadVariableOpReadVariableOp>dunet_brats_decathlon_decodeaa_biasadd_readvariableop_resource*
_output_shapes
: *
dtype028
62DUNet_Brats_Decathlon/decodeAa/BiasAdd/ReadVariableOp�
'2DUNet_Brats_Decathlon/decodeAa/BiasAddBiasAdd/2DUNet_Brats_Decathlon/decodeAa/Conv2D:output:0>2DUNet_Brats_Decathlon/decodeAa/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@ 2)
'2DUNet_Brats_Decathlon/decodeAa/BiasAdd�
$2DUNet_Brats_Decathlon/decodeAa/ReluRelu02DUNet_Brats_Decathlon/decodeAa/BiasAdd:output:0*
T0*/
_output_shapes
:���������@@ 2&
$2DUNet_Brats_Decathlon/decodeAa/Relu�
52DUNet_Brats_Decathlon/decodeAb/Conv2D/ReadVariableOpReadVariableOp=dunet_brats_decathlon_decodeab_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype027
52DUNet_Brats_Decathlon/decodeAb/Conv2D/ReadVariableOp�
&2DUNet_Brats_Decathlon/decodeAb/Conv2DConv2D22DUNet_Brats_Decathlon/decodeAa/Relu:activations:0=2DUNet_Brats_Decathlon/decodeAb/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@ *
paddingSAME*
strides
2(
&2DUNet_Brats_Decathlon/decodeAb/Conv2D�
62DUNet_Brats_Decathlon/decodeAb/BiasAdd/ReadVariableOpReadVariableOp>dunet_brats_decathlon_decodeab_biasadd_readvariableop_resource*
_output_shapes
: *
dtype028
62DUNet_Brats_Decathlon/decodeAb/BiasAdd/ReadVariableOp�
'2DUNet_Brats_Decathlon/decodeAb/BiasAddBiasAdd/2DUNet_Brats_Decathlon/decodeAb/Conv2D:output:0>2DUNet_Brats_Decathlon/decodeAb/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@ 2)
'2DUNet_Brats_Decathlon/decodeAb/BiasAdd�
$2DUNet_Brats_Decathlon/decodeAb/ReluRelu02DUNet_Brats_Decathlon/decodeAb/BiasAdd:output:0*
T0*/
_output_shapes
:���������@@ 2&
$2DUNet_Brats_Decathlon/decodeAb/Relu�
'2DUNet_Brats_Decathlon/transconvA/ShapeShape22DUNet_Brats_Decathlon/decodeAb/Relu:activations:0*
T0*
_output_shapes
:2)
'2DUNet_Brats_Decathlon/transconvA/Shape�
52DUNet_Brats_Decathlon/transconvA/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
52DUNet_Brats_Decathlon/transconvA/strided_slice/stack�
72DUNet_Brats_Decathlon/transconvA/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
72DUNet_Brats_Decathlon/transconvA/strided_slice/stack_1�
72DUNet_Brats_Decathlon/transconvA/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
72DUNet_Brats_Decathlon/transconvA/strided_slice/stack_2�
/2DUNet_Brats_Decathlon/transconvA/strided_sliceStridedSlice02DUNet_Brats_Decathlon/transconvA/Shape:output:0>2DUNet_Brats_Decathlon/transconvA/strided_slice/stack:output:0@2DUNet_Brats_Decathlon/transconvA/strided_slice/stack_1:output:0@2DUNet_Brats_Decathlon/transconvA/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/2DUNet_Brats_Decathlon/transconvA/strided_slice�
)2DUNet_Brats_Decathlon/transconvA/stack/1Const*
_output_shapes
: *
dtype0*
value
B :�2+
)2DUNet_Brats_Decathlon/transconvA/stack/1�
)2DUNet_Brats_Decathlon/transconvA/stack/2Const*
_output_shapes
: *
dtype0*
value
B :�2+
)2DUNet_Brats_Decathlon/transconvA/stack/2�
)2DUNet_Brats_Decathlon/transconvA/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2+
)2DUNet_Brats_Decathlon/transconvA/stack/3�
'2DUNet_Brats_Decathlon/transconvA/stackPack82DUNet_Brats_Decathlon/transconvA/strided_slice:output:022DUNet_Brats_Decathlon/transconvA/stack/1:output:022DUNet_Brats_Decathlon/transconvA/stack/2:output:022DUNet_Brats_Decathlon/transconvA/stack/3:output:0*
N*
T0*
_output_shapes
:2)
'2DUNet_Brats_Decathlon/transconvA/stack�
72DUNet_Brats_Decathlon/transconvA/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 29
72DUNet_Brats_Decathlon/transconvA/strided_slice_1/stack�
92DUNet_Brats_Decathlon/transconvA/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
92DUNet_Brats_Decathlon/transconvA/strided_slice_1/stack_1�
92DUNet_Brats_Decathlon/transconvA/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
92DUNet_Brats_Decathlon/transconvA/strided_slice_1/stack_2�
12DUNet_Brats_Decathlon/transconvA/strided_slice_1StridedSlice02DUNet_Brats_Decathlon/transconvA/stack:output:0@2DUNet_Brats_Decathlon/transconvA/strided_slice_1/stack:output:0B2DUNet_Brats_Decathlon/transconvA/strided_slice_1/stack_1:output:0B2DUNet_Brats_Decathlon/transconvA/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask23
12DUNet_Brats_Decathlon/transconvA/strided_slice_1�
A2DUNet_Brats_Decathlon/transconvA/conv2d_transpose/ReadVariableOpReadVariableOpIdunet_brats_decathlon_transconva_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02C
A2DUNet_Brats_Decathlon/transconvA/conv2d_transpose/ReadVariableOp�
22DUNet_Brats_Decathlon/transconvA/conv2d_transposeConv2DBackpropInput02DUNet_Brats_Decathlon/transconvA/stack:output:0I2DUNet_Brats_Decathlon/transconvA/conv2d_transpose/ReadVariableOp:value:022DUNet_Brats_Decathlon/decodeAb/Relu:activations:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
24
22DUNet_Brats_Decathlon/transconvA/conv2d_transpose�
82DUNet_Brats_Decathlon/transconvA/BiasAdd/ReadVariableOpReadVariableOp@dunet_brats_decathlon_transconva_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02:
82DUNet_Brats_Decathlon/transconvA/BiasAdd/ReadVariableOp�
)2DUNet_Brats_Decathlon/transconvA/BiasAddBiasAdd;2DUNet_Brats_Decathlon/transconvA/conv2d_transpose:output:0@2DUNet_Brats_Decathlon/transconvA/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������2+
)2DUNet_Brats_Decathlon/transconvA/BiasAdd�
*2DUNet_Brats_Decathlon/concatA/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2,
*2DUNet_Brats_Decathlon/concatA/concat/axis�
%2DUNet_Brats_Decathlon/concatA/concatConcatV222DUNet_Brats_Decathlon/transconvA/BiasAdd:output:022DUNet_Brats_Decathlon/encodeAb/Relu:activations:032DUNet_Brats_Decathlon/concatA/concat/axis:output:0*
N*
T0*1
_output_shapes
:����������� 2'
%2DUNet_Brats_Decathlon/concatA/concat�
52DUNet_Brats_Decathlon/convOuta/Conv2D/ReadVariableOpReadVariableOp=dunet_brats_decathlon_convouta_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype027
52DUNet_Brats_Decathlon/convOuta/Conv2D/ReadVariableOp�
&2DUNet_Brats_Decathlon/convOuta/Conv2DConv2D.2DUNet_Brats_Decathlon/concatA/concat:output:0=2DUNet_Brats_Decathlon/convOuta/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
2(
&2DUNet_Brats_Decathlon/convOuta/Conv2D�
62DUNet_Brats_Decathlon/convOuta/BiasAdd/ReadVariableOpReadVariableOp>dunet_brats_decathlon_convouta_biasadd_readvariableop_resource*
_output_shapes
:*
dtype028
62DUNet_Brats_Decathlon/convOuta/BiasAdd/ReadVariableOp�
'2DUNet_Brats_Decathlon/convOuta/BiasAddBiasAdd/2DUNet_Brats_Decathlon/convOuta/Conv2D:output:0>2DUNet_Brats_Decathlon/convOuta/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������2)
'2DUNet_Brats_Decathlon/convOuta/BiasAdd�
$2DUNet_Brats_Decathlon/convOuta/ReluRelu02DUNet_Brats_Decathlon/convOuta/BiasAdd:output:0*
T0*1
_output_shapes
:�����������2&
$2DUNet_Brats_Decathlon/convOuta/Relu�
52DUNet_Brats_Decathlon/convOutb/Conv2D/ReadVariableOpReadVariableOp=dunet_brats_decathlon_convoutb_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype027
52DUNet_Brats_Decathlon/convOutb/Conv2D/ReadVariableOp�
&2DUNet_Brats_Decathlon/convOutb/Conv2DConv2D22DUNet_Brats_Decathlon/convOuta/Relu:activations:0=2DUNet_Brats_Decathlon/convOutb/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
2(
&2DUNet_Brats_Decathlon/convOutb/Conv2D�
62DUNet_Brats_Decathlon/convOutb/BiasAdd/ReadVariableOpReadVariableOp>dunet_brats_decathlon_convoutb_biasadd_readvariableop_resource*
_output_shapes
:*
dtype028
62DUNet_Brats_Decathlon/convOutb/BiasAdd/ReadVariableOp�
'2DUNet_Brats_Decathlon/convOutb/BiasAddBiasAdd/2DUNet_Brats_Decathlon/convOutb/Conv2D:output:0>2DUNet_Brats_Decathlon/convOutb/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������2)
'2DUNet_Brats_Decathlon/convOutb/BiasAdd�
$2DUNet_Brats_Decathlon/convOutb/ReluRelu02DUNet_Brats_Decathlon/convOutb/BiasAdd:output:0*
T0*1
_output_shapes
:�����������2&
$2DUNet_Brats_Decathlon/convOutb/Relu�
;2DUNet_Brats_Decathlon/PredictionMask/Conv2D/ReadVariableOpReadVariableOpCdunet_brats_decathlon_predictionmask_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02=
;2DUNet_Brats_Decathlon/PredictionMask/Conv2D/ReadVariableOp�
,2DUNet_Brats_Decathlon/PredictionMask/Conv2DConv2D22DUNet_Brats_Decathlon/convOutb/Relu:activations:0C2DUNet_Brats_Decathlon/PredictionMask/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������*
paddingVALID*
strides
2.
,2DUNet_Brats_Decathlon/PredictionMask/Conv2D�
<2DUNet_Brats_Decathlon/PredictionMask/BiasAdd/ReadVariableOpReadVariableOpDdunet_brats_decathlon_predictionmask_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02>
<2DUNet_Brats_Decathlon/PredictionMask/BiasAdd/ReadVariableOp�
-2DUNet_Brats_Decathlon/PredictionMask/BiasAddBiasAdd52DUNet_Brats_Decathlon/PredictionMask/Conv2D:output:0D2DUNet_Brats_Decathlon/PredictionMask/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������2/
-2DUNet_Brats_Decathlon/PredictionMask/BiasAdd�
-2DUNet_Brats_Decathlon/PredictionMask/SigmoidSigmoid62DUNet_Brats_Decathlon/PredictionMask/BiasAdd:output:0*
T0*1
_output_shapes
:�����������2/
-2DUNet_Brats_Decathlon/PredictionMask/Sigmoid�
IdentityIdentity12DUNet_Brats_Decathlon/PredictionMask/Sigmoid:y:0=^2DUNet_Brats_Decathlon/PredictionMask/BiasAdd/ReadVariableOp<^2DUNet_Brats_Decathlon/PredictionMask/Conv2D/ReadVariableOp7^2DUNet_Brats_Decathlon/convOuta/BiasAdd/ReadVariableOp6^2DUNet_Brats_Decathlon/convOuta/Conv2D/ReadVariableOp7^2DUNet_Brats_Decathlon/convOutb/BiasAdd/ReadVariableOp6^2DUNet_Brats_Decathlon/convOutb/Conv2D/ReadVariableOp7^2DUNet_Brats_Decathlon/decodeAa/BiasAdd/ReadVariableOp6^2DUNet_Brats_Decathlon/decodeAa/Conv2D/ReadVariableOp7^2DUNet_Brats_Decathlon/decodeAb/BiasAdd/ReadVariableOp6^2DUNet_Brats_Decathlon/decodeAb/Conv2D/ReadVariableOp7^2DUNet_Brats_Decathlon/decodeBa/BiasAdd/ReadVariableOp6^2DUNet_Brats_Decathlon/decodeBa/Conv2D/ReadVariableOp7^2DUNet_Brats_Decathlon/decodeBb/BiasAdd/ReadVariableOp6^2DUNet_Brats_Decathlon/decodeBb/Conv2D/ReadVariableOp7^2DUNet_Brats_Decathlon/decodeCa/BiasAdd/ReadVariableOp6^2DUNet_Brats_Decathlon/decodeCa/Conv2D/ReadVariableOp7^2DUNet_Brats_Decathlon/decodeCb/BiasAdd/ReadVariableOp6^2DUNet_Brats_Decathlon/decodeCb/Conv2D/ReadVariableOp7^2DUNet_Brats_Decathlon/encodeAa/BiasAdd/ReadVariableOp6^2DUNet_Brats_Decathlon/encodeAa/Conv2D/ReadVariableOp7^2DUNet_Brats_Decathlon/encodeAb/BiasAdd/ReadVariableOp6^2DUNet_Brats_Decathlon/encodeAb/Conv2D/ReadVariableOp7^2DUNet_Brats_Decathlon/encodeBa/BiasAdd/ReadVariableOp6^2DUNet_Brats_Decathlon/encodeBa/Conv2D/ReadVariableOp7^2DUNet_Brats_Decathlon/encodeBb/BiasAdd/ReadVariableOp6^2DUNet_Brats_Decathlon/encodeBb/Conv2D/ReadVariableOp7^2DUNet_Brats_Decathlon/encodeCa/BiasAdd/ReadVariableOp6^2DUNet_Brats_Decathlon/encodeCa/Conv2D/ReadVariableOp7^2DUNet_Brats_Decathlon/encodeCb/BiasAdd/ReadVariableOp6^2DUNet_Brats_Decathlon/encodeCb/Conv2D/ReadVariableOp7^2DUNet_Brats_Decathlon/encodeDa/BiasAdd/ReadVariableOp6^2DUNet_Brats_Decathlon/encodeDa/Conv2D/ReadVariableOp7^2DUNet_Brats_Decathlon/encodeDb/BiasAdd/ReadVariableOp6^2DUNet_Brats_Decathlon/encodeDb/Conv2D/ReadVariableOp7^2DUNet_Brats_Decathlon/encodeEa/BiasAdd/ReadVariableOp6^2DUNet_Brats_Decathlon/encodeEa/Conv2D/ReadVariableOp7^2DUNet_Brats_Decathlon/encodeEb/BiasAdd/ReadVariableOp6^2DUNet_Brats_Decathlon/encodeEb/Conv2D/ReadVariableOp9^2DUNet_Brats_Decathlon/transconvA/BiasAdd/ReadVariableOpB^2DUNet_Brats_Decathlon/transconvA/conv2d_transpose/ReadVariableOp9^2DUNet_Brats_Decathlon/transconvB/BiasAdd/ReadVariableOpB^2DUNet_Brats_Decathlon/transconvB/conv2d_transpose/ReadVariableOp9^2DUNet_Brats_Decathlon/transconvC/BiasAdd/ReadVariableOpB^2DUNet_Brats_Decathlon/transconvC/conv2d_transpose/ReadVariableOp9^2DUNet_Brats_Decathlon/transconvE/BiasAdd/ReadVariableOpB^2DUNet_Brats_Decathlon/transconvE/conv2d_transpose/ReadVariableOp*
T0*1
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:�����������::::::::::::::::::::::::::::::::::::::::::::::2|
<2DUNet_Brats_Decathlon/PredictionMask/BiasAdd/ReadVariableOp<2DUNet_Brats_Decathlon/PredictionMask/BiasAdd/ReadVariableOp2z
;2DUNet_Brats_Decathlon/PredictionMask/Conv2D/ReadVariableOp;2DUNet_Brats_Decathlon/PredictionMask/Conv2D/ReadVariableOp2p
62DUNet_Brats_Decathlon/convOuta/BiasAdd/ReadVariableOp62DUNet_Brats_Decathlon/convOuta/BiasAdd/ReadVariableOp2n
52DUNet_Brats_Decathlon/convOuta/Conv2D/ReadVariableOp52DUNet_Brats_Decathlon/convOuta/Conv2D/ReadVariableOp2p
62DUNet_Brats_Decathlon/convOutb/BiasAdd/ReadVariableOp62DUNet_Brats_Decathlon/convOutb/BiasAdd/ReadVariableOp2n
52DUNet_Brats_Decathlon/convOutb/Conv2D/ReadVariableOp52DUNet_Brats_Decathlon/convOutb/Conv2D/ReadVariableOp2p
62DUNet_Brats_Decathlon/decodeAa/BiasAdd/ReadVariableOp62DUNet_Brats_Decathlon/decodeAa/BiasAdd/ReadVariableOp2n
52DUNet_Brats_Decathlon/decodeAa/Conv2D/ReadVariableOp52DUNet_Brats_Decathlon/decodeAa/Conv2D/ReadVariableOp2p
62DUNet_Brats_Decathlon/decodeAb/BiasAdd/ReadVariableOp62DUNet_Brats_Decathlon/decodeAb/BiasAdd/ReadVariableOp2n
52DUNet_Brats_Decathlon/decodeAb/Conv2D/ReadVariableOp52DUNet_Brats_Decathlon/decodeAb/Conv2D/ReadVariableOp2p
62DUNet_Brats_Decathlon/decodeBa/BiasAdd/ReadVariableOp62DUNet_Brats_Decathlon/decodeBa/BiasAdd/ReadVariableOp2n
52DUNet_Brats_Decathlon/decodeBa/Conv2D/ReadVariableOp52DUNet_Brats_Decathlon/decodeBa/Conv2D/ReadVariableOp2p
62DUNet_Brats_Decathlon/decodeBb/BiasAdd/ReadVariableOp62DUNet_Brats_Decathlon/decodeBb/BiasAdd/ReadVariableOp2n
52DUNet_Brats_Decathlon/decodeBb/Conv2D/ReadVariableOp52DUNet_Brats_Decathlon/decodeBb/Conv2D/ReadVariableOp2p
62DUNet_Brats_Decathlon/decodeCa/BiasAdd/ReadVariableOp62DUNet_Brats_Decathlon/decodeCa/BiasAdd/ReadVariableOp2n
52DUNet_Brats_Decathlon/decodeCa/Conv2D/ReadVariableOp52DUNet_Brats_Decathlon/decodeCa/Conv2D/ReadVariableOp2p
62DUNet_Brats_Decathlon/decodeCb/BiasAdd/ReadVariableOp62DUNet_Brats_Decathlon/decodeCb/BiasAdd/ReadVariableOp2n
52DUNet_Brats_Decathlon/decodeCb/Conv2D/ReadVariableOp52DUNet_Brats_Decathlon/decodeCb/Conv2D/ReadVariableOp2p
62DUNet_Brats_Decathlon/encodeAa/BiasAdd/ReadVariableOp62DUNet_Brats_Decathlon/encodeAa/BiasAdd/ReadVariableOp2n
52DUNet_Brats_Decathlon/encodeAa/Conv2D/ReadVariableOp52DUNet_Brats_Decathlon/encodeAa/Conv2D/ReadVariableOp2p
62DUNet_Brats_Decathlon/encodeAb/BiasAdd/ReadVariableOp62DUNet_Brats_Decathlon/encodeAb/BiasAdd/ReadVariableOp2n
52DUNet_Brats_Decathlon/encodeAb/Conv2D/ReadVariableOp52DUNet_Brats_Decathlon/encodeAb/Conv2D/ReadVariableOp2p
62DUNet_Brats_Decathlon/encodeBa/BiasAdd/ReadVariableOp62DUNet_Brats_Decathlon/encodeBa/BiasAdd/ReadVariableOp2n
52DUNet_Brats_Decathlon/encodeBa/Conv2D/ReadVariableOp52DUNet_Brats_Decathlon/encodeBa/Conv2D/ReadVariableOp2p
62DUNet_Brats_Decathlon/encodeBb/BiasAdd/ReadVariableOp62DUNet_Brats_Decathlon/encodeBb/BiasAdd/ReadVariableOp2n
52DUNet_Brats_Decathlon/encodeBb/Conv2D/ReadVariableOp52DUNet_Brats_Decathlon/encodeBb/Conv2D/ReadVariableOp2p
62DUNet_Brats_Decathlon/encodeCa/BiasAdd/ReadVariableOp62DUNet_Brats_Decathlon/encodeCa/BiasAdd/ReadVariableOp2n
52DUNet_Brats_Decathlon/encodeCa/Conv2D/ReadVariableOp52DUNet_Brats_Decathlon/encodeCa/Conv2D/ReadVariableOp2p
62DUNet_Brats_Decathlon/encodeCb/BiasAdd/ReadVariableOp62DUNet_Brats_Decathlon/encodeCb/BiasAdd/ReadVariableOp2n
52DUNet_Brats_Decathlon/encodeCb/Conv2D/ReadVariableOp52DUNet_Brats_Decathlon/encodeCb/Conv2D/ReadVariableOp2p
62DUNet_Brats_Decathlon/encodeDa/BiasAdd/ReadVariableOp62DUNet_Brats_Decathlon/encodeDa/BiasAdd/ReadVariableOp2n
52DUNet_Brats_Decathlon/encodeDa/Conv2D/ReadVariableOp52DUNet_Brats_Decathlon/encodeDa/Conv2D/ReadVariableOp2p
62DUNet_Brats_Decathlon/encodeDb/BiasAdd/ReadVariableOp62DUNet_Brats_Decathlon/encodeDb/BiasAdd/ReadVariableOp2n
52DUNet_Brats_Decathlon/encodeDb/Conv2D/ReadVariableOp52DUNet_Brats_Decathlon/encodeDb/Conv2D/ReadVariableOp2p
62DUNet_Brats_Decathlon/encodeEa/BiasAdd/ReadVariableOp62DUNet_Brats_Decathlon/encodeEa/BiasAdd/ReadVariableOp2n
52DUNet_Brats_Decathlon/encodeEa/Conv2D/ReadVariableOp52DUNet_Brats_Decathlon/encodeEa/Conv2D/ReadVariableOp2p
62DUNet_Brats_Decathlon/encodeEb/BiasAdd/ReadVariableOp62DUNet_Brats_Decathlon/encodeEb/BiasAdd/ReadVariableOp2n
52DUNet_Brats_Decathlon/encodeEb/Conv2D/ReadVariableOp52DUNet_Brats_Decathlon/encodeEb/Conv2D/ReadVariableOp2t
82DUNet_Brats_Decathlon/transconvA/BiasAdd/ReadVariableOp82DUNet_Brats_Decathlon/transconvA/BiasAdd/ReadVariableOp2�
A2DUNet_Brats_Decathlon/transconvA/conv2d_transpose/ReadVariableOpA2DUNet_Brats_Decathlon/transconvA/conv2d_transpose/ReadVariableOp2t
82DUNet_Brats_Decathlon/transconvB/BiasAdd/ReadVariableOp82DUNet_Brats_Decathlon/transconvB/BiasAdd/ReadVariableOp2�
A2DUNet_Brats_Decathlon/transconvB/conv2d_transpose/ReadVariableOpA2DUNet_Brats_Decathlon/transconvB/conv2d_transpose/ReadVariableOp2t
82DUNet_Brats_Decathlon/transconvC/BiasAdd/ReadVariableOp82DUNet_Brats_Decathlon/transconvC/BiasAdd/ReadVariableOp2�
A2DUNet_Brats_Decathlon/transconvC/conv2d_transpose/ReadVariableOpA2DUNet_Brats_Decathlon/transconvC/conv2d_transpose/ReadVariableOp2t
82DUNet_Brats_Decathlon/transconvE/BiasAdd/ReadVariableOp82DUNet_Brats_Decathlon/transconvE/BiasAdd/ReadVariableOp2�
A2DUNet_Brats_Decathlon/transconvE/conv2d_transpose/ReadVariableOpA2DUNet_Brats_Decathlon/transconvE/conv2d_transpose/ReadVariableOp:[ W
1
_output_shapes
:�����������
"
_user_specified_name
MRImages
�
l
N__inference_spatial_dropout2d_1_layer_call_and_return_conditional_losses_18593

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:����������2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:����������2

Identity_1"!

identity_1Identity_1:output:0*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
l
N__inference_spatial_dropout2d_1_layer_call_and_return_conditional_losses_20670

inputs

identity_1}
IdentityIdentityinputs*
T0*J
_output_shapes8
6:4������������������������������������2

Identity�

Identity_1IdentityIdentity:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity_1"!

identity_1Identity_1:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
}
(__inference_convOutb_layer_call_fn_20952

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_convOutb_layer_call_and_return_conditional_losses_189712
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:�����������::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
l
B__inference_concatA_layer_call_and_return_conditional_losses_18924

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis�
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*1
_output_shapes
:����������� 2
concatm
IdentityIdentityconcat:output:0*
T0*1
_output_shapes
:����������� 2

Identity"
identityIdentity:output:0*]
_input_shapesL
J:+���������������������������:�����������:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs:YU
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�

�
C__inference_decodeCa_layer_call_and_return_conditional_losses_18719

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:����������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
C__inference_encodeBa_layer_call_and_return_conditional_losses_18401

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@ *
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@ 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@@ 2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:���������@@ 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������@@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������@@
 
_user_specified_nameinputs
�
}
(__inference_encodeBa_layer_call_fn_20448

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_encodeBa_layer_call_and_return_conditional_losses_184012
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������@@ 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������@@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������@@
 
_user_specified_nameinputs
�
\
@__inference_poolA_layer_call_and_return_conditional_losses_17977

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
2	
MaxPool�
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�

�
C__inference_decodeCa_layer_call_and_return_conditional_losses_20764

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:����������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
6__inference_2DUNet_Brats_Decathlon_layer_call_fn_19597
mrimages
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallmrimagesunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44*:
Tin3
12/*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*P
_read_only_resource_inputs2
0.	
 !"#$%&'()*+,-.*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_2DUNet_Brats_Decathlon_layer_call_and_return_conditional_losses_195022
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:�����������::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:[ W
1
_output_shapes
:�����������
"
_user_specified_name
MRImages
�

*__inference_transconvC_layer_call_fn_18243

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_transconvC_layer_call_and_return_conditional_losses_182332
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,����������������������������::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�

�
C__inference_encodeCb_layer_call_and_return_conditional_losses_20575

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  @*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  @2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������  @2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:���������  @2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������  @::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������  @
 
_user_specified_nameinputs
�
�
Q__inference_2DUNet_Brats_Decathlon_layer_call_and_return_conditional_losses_19015
mrimages
encodeaa_18357
encodeaa_18359
encodeab_18384
encodeab_18386
encodeba_18412
encodeba_18414
encodebb_18439
encodebb_18441
encodeca_18467
encodeca_18469
encodecb_18533
encodecb_18535
encodeda_18561
encodeda_18563
encodedb_18627
encodedb_18629
encodeea_18655
encodeea_18657
encodeeb_18682
encodeeb_18684
transconve_18687
transconve_18689
decodeca_18730
decodeca_18732
decodecb_18757
decodecb_18759
transconvc_18762
transconvc_18764
decodeba_18805
decodeba_18807
decodebb_18832
decodebb_18834
transconvb_18837
transconvb_18839
decodeaa_18880
decodeaa_18882
decodeab_18907
decodeab_18909
transconva_18912
transconva_18914
convouta_18955
convouta_18957
convoutb_18982
convoutb_18984
predictionmask_19009
predictionmask_19011
identity��&PredictionMask/StatefulPartitionedCall� convOuta/StatefulPartitionedCall� convOutb/StatefulPartitionedCall� decodeAa/StatefulPartitionedCall� decodeAb/StatefulPartitionedCall� decodeBa/StatefulPartitionedCall� decodeBb/StatefulPartitionedCall� decodeCa/StatefulPartitionedCall� decodeCb/StatefulPartitionedCall� encodeAa/StatefulPartitionedCall� encodeAb/StatefulPartitionedCall� encodeBa/StatefulPartitionedCall� encodeBb/StatefulPartitionedCall� encodeCa/StatefulPartitionedCall� encodeCb/StatefulPartitionedCall� encodeDa/StatefulPartitionedCall� encodeDb/StatefulPartitionedCall� encodeEa/StatefulPartitionedCall� encodeEb/StatefulPartitionedCall�)spatial_dropout2d/StatefulPartitionedCall�+spatial_dropout2d_1/StatefulPartitionedCall�"transconvA/StatefulPartitionedCall�"transconvB/StatefulPartitionedCall�"transconvC/StatefulPartitionedCall�"transconvE/StatefulPartitionedCall�
 encodeAa/StatefulPartitionedCallStatefulPartitionedCallmrimagesencodeaa_18357encodeaa_18359*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_encodeAa_layer_call_and_return_conditional_losses_183462"
 encodeAa/StatefulPartitionedCall�
 encodeAb/StatefulPartitionedCallStatefulPartitionedCall)encodeAa/StatefulPartitionedCall:output:0encodeab_18384encodeab_18386*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_encodeAb_layer_call_and_return_conditional_losses_183732"
 encodeAb/StatefulPartitionedCall�
poolA/PartitionedCallPartitionedCall)encodeAb/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_poolA_layer_call_and_return_conditional_losses_179772
poolA/PartitionedCall�
 encodeBa/StatefulPartitionedCallStatefulPartitionedCallpoolA/PartitionedCall:output:0encodeba_18412encodeba_18414*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_encodeBa_layer_call_and_return_conditional_losses_184012"
 encodeBa/StatefulPartitionedCall�
 encodeBb/StatefulPartitionedCallStatefulPartitionedCall)encodeBa/StatefulPartitionedCall:output:0encodebb_18439encodebb_18441*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_encodeBb_layer_call_and_return_conditional_losses_184282"
 encodeBb/StatefulPartitionedCall�
poolB/PartitionedCallPartitionedCall)encodeBb/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������   * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_poolB_layer_call_and_return_conditional_losses_179892
poolB/PartitionedCall�
 encodeCa/StatefulPartitionedCallStatefulPartitionedCallpoolB/PartitionedCall:output:0encodeca_18467encodeca_18469*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_encodeCa_layer_call_and_return_conditional_losses_184562"
 encodeCa/StatefulPartitionedCall�
)spatial_dropout2d/StatefulPartitionedCallStatefulPartitionedCall)encodeCa/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_spatial_dropout2d_layer_call_and_return_conditional_losses_184942+
)spatial_dropout2d/StatefulPartitionedCall�
 encodeCb/StatefulPartitionedCallStatefulPartitionedCall2spatial_dropout2d/StatefulPartitionedCall:output:0encodecb_18533encodecb_18535*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_encodeCb_layer_call_and_return_conditional_losses_185222"
 encodeCb/StatefulPartitionedCall�
poolC/PartitionedCallPartitionedCall)encodeCb/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_poolC_layer_call_and_return_conditional_losses_180692
poolC/PartitionedCall�
 encodeDa/StatefulPartitionedCallStatefulPartitionedCallpoolC/PartitionedCall:output:0encodeda_18561encodeda_18563*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_encodeDa_layer_call_and_return_conditional_losses_185502"
 encodeDa/StatefulPartitionedCall�
+spatial_dropout2d_1/StatefulPartitionedCallStatefulPartitionedCall)encodeDa/StatefulPartitionedCall:output:0*^spatial_dropout2d/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_spatial_dropout2d_1_layer_call_and_return_conditional_losses_185882-
+spatial_dropout2d_1/StatefulPartitionedCall�
 encodeDb/StatefulPartitionedCallStatefulPartitionedCall4spatial_dropout2d_1/StatefulPartitionedCall:output:0encodedb_18627encodedb_18629*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_encodeDb_layer_call_and_return_conditional_losses_186162"
 encodeDb/StatefulPartitionedCall�
poolD/PartitionedCallPartitionedCall)encodeDb/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_poolD_layer_call_and_return_conditional_losses_181492
poolD/PartitionedCall�
 encodeEa/StatefulPartitionedCallStatefulPartitionedCallpoolD/PartitionedCall:output:0encodeea_18655encodeea_18657*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_encodeEa_layer_call_and_return_conditional_losses_186442"
 encodeEa/StatefulPartitionedCall�
 encodeEb/StatefulPartitionedCallStatefulPartitionedCall)encodeEa/StatefulPartitionedCall:output:0encodeeb_18682encodeeb_18684*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_encodeEb_layer_call_and_return_conditional_losses_186712"
 encodeEb/StatefulPartitionedCall�
"transconvE/StatefulPartitionedCallStatefulPartitionedCall)encodeEb/StatefulPartitionedCall:output:0transconve_18687transconve_18689*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_transconvE_layer_call_and_return_conditional_losses_181892$
"transconvE/StatefulPartitionedCall�
concatD/PartitionedCallPartitionedCall+transconvE/StatefulPartitionedCall:output:0)encodeDb/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_concatD_layer_call_and_return_conditional_losses_186992
concatD/PartitionedCall�
 decodeCa/StatefulPartitionedCallStatefulPartitionedCall concatD/PartitionedCall:output:0decodeca_18730decodeca_18732*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_decodeCa_layer_call_and_return_conditional_losses_187192"
 decodeCa/StatefulPartitionedCall�
 decodeCb/StatefulPartitionedCallStatefulPartitionedCall)decodeCa/StatefulPartitionedCall:output:0decodecb_18757decodecb_18759*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_decodeCb_layer_call_and_return_conditional_losses_187462"
 decodeCb/StatefulPartitionedCall�
"transconvC/StatefulPartitionedCallStatefulPartitionedCall)decodeCb/StatefulPartitionedCall:output:0transconvc_18762transconvc_18764*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_transconvC_layer_call_and_return_conditional_losses_182332$
"transconvC/StatefulPartitionedCall�
concatC/PartitionedCallPartitionedCall+transconvC/StatefulPartitionedCall:output:0)encodeCb/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������  �* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_concatC_layer_call_and_return_conditional_losses_187742
concatC/PartitionedCall�
 decodeBa/StatefulPartitionedCallStatefulPartitionedCall concatC/PartitionedCall:output:0decodeba_18805decodeba_18807*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_decodeBa_layer_call_and_return_conditional_losses_187942"
 decodeBa/StatefulPartitionedCall�
 decodeBb/StatefulPartitionedCallStatefulPartitionedCall)decodeBa/StatefulPartitionedCall:output:0decodebb_18832decodebb_18834*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_decodeBb_layer_call_and_return_conditional_losses_188212"
 decodeBb/StatefulPartitionedCall�
"transconvB/StatefulPartitionedCallStatefulPartitionedCall)decodeBb/StatefulPartitionedCall:output:0transconvb_18837transconvb_18839*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+��������������������������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_transconvB_layer_call_and_return_conditional_losses_182772$
"transconvB/StatefulPartitionedCall�
concatB/PartitionedCallPartitionedCall+transconvB/StatefulPartitionedCall:output:0)encodeBb/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_concatB_layer_call_and_return_conditional_losses_188492
concatB/PartitionedCall�
 decodeAa/StatefulPartitionedCallStatefulPartitionedCall concatB/PartitionedCall:output:0decodeaa_18880decodeaa_18882*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_decodeAa_layer_call_and_return_conditional_losses_188692"
 decodeAa/StatefulPartitionedCall�
 decodeAb/StatefulPartitionedCallStatefulPartitionedCall)decodeAa/StatefulPartitionedCall:output:0decodeab_18907decodeab_18909*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_decodeAb_layer_call_and_return_conditional_losses_188962"
 decodeAb/StatefulPartitionedCall�
"transconvA/StatefulPartitionedCallStatefulPartitionedCall)decodeAb/StatefulPartitionedCall:output:0transconva_18912transconva_18914*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_transconvA_layer_call_and_return_conditional_losses_183212$
"transconvA/StatefulPartitionedCall�
concatA/PartitionedCallPartitionedCall+transconvA/StatefulPartitionedCall:output:0)encodeAb/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:����������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_concatA_layer_call_and_return_conditional_losses_189242
concatA/PartitionedCall�
 convOuta/StatefulPartitionedCallStatefulPartitionedCall concatA/PartitionedCall:output:0convouta_18955convouta_18957*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_convOuta_layer_call_and_return_conditional_losses_189442"
 convOuta/StatefulPartitionedCall�
 convOutb/StatefulPartitionedCallStatefulPartitionedCall)convOuta/StatefulPartitionedCall:output:0convoutb_18982convoutb_18984*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_convOutb_layer_call_and_return_conditional_losses_189712"
 convOutb/StatefulPartitionedCall�
&PredictionMask/StatefulPartitionedCallStatefulPartitionedCall)convOutb/StatefulPartitionedCall:output:0predictionmask_19009predictionmask_19011*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_PredictionMask_layer_call_and_return_conditional_losses_189982(
&PredictionMask/StatefulPartitionedCall�
IdentityIdentity/PredictionMask/StatefulPartitionedCall:output:0'^PredictionMask/StatefulPartitionedCall!^convOuta/StatefulPartitionedCall!^convOutb/StatefulPartitionedCall!^decodeAa/StatefulPartitionedCall!^decodeAb/StatefulPartitionedCall!^decodeBa/StatefulPartitionedCall!^decodeBb/StatefulPartitionedCall!^decodeCa/StatefulPartitionedCall!^decodeCb/StatefulPartitionedCall!^encodeAa/StatefulPartitionedCall!^encodeAb/StatefulPartitionedCall!^encodeBa/StatefulPartitionedCall!^encodeBb/StatefulPartitionedCall!^encodeCa/StatefulPartitionedCall!^encodeCb/StatefulPartitionedCall!^encodeDa/StatefulPartitionedCall!^encodeDb/StatefulPartitionedCall!^encodeEa/StatefulPartitionedCall!^encodeEb/StatefulPartitionedCall*^spatial_dropout2d/StatefulPartitionedCall,^spatial_dropout2d_1/StatefulPartitionedCall#^transconvA/StatefulPartitionedCall#^transconvB/StatefulPartitionedCall#^transconvC/StatefulPartitionedCall#^transconvE/StatefulPartitionedCall*
T0*1
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:�����������::::::::::::::::::::::::::::::::::::::::::::::2P
&PredictionMask/StatefulPartitionedCall&PredictionMask/StatefulPartitionedCall2D
 convOuta/StatefulPartitionedCall convOuta/StatefulPartitionedCall2D
 convOutb/StatefulPartitionedCall convOutb/StatefulPartitionedCall2D
 decodeAa/StatefulPartitionedCall decodeAa/StatefulPartitionedCall2D
 decodeAb/StatefulPartitionedCall decodeAb/StatefulPartitionedCall2D
 decodeBa/StatefulPartitionedCall decodeBa/StatefulPartitionedCall2D
 decodeBb/StatefulPartitionedCall decodeBb/StatefulPartitionedCall2D
 decodeCa/StatefulPartitionedCall decodeCa/StatefulPartitionedCall2D
 decodeCb/StatefulPartitionedCall decodeCb/StatefulPartitionedCall2D
 encodeAa/StatefulPartitionedCall encodeAa/StatefulPartitionedCall2D
 encodeAb/StatefulPartitionedCall encodeAb/StatefulPartitionedCall2D
 encodeBa/StatefulPartitionedCall encodeBa/StatefulPartitionedCall2D
 encodeBb/StatefulPartitionedCall encodeBb/StatefulPartitionedCall2D
 encodeCa/StatefulPartitionedCall encodeCa/StatefulPartitionedCall2D
 encodeCb/StatefulPartitionedCall encodeCb/StatefulPartitionedCall2D
 encodeDa/StatefulPartitionedCall encodeDa/StatefulPartitionedCall2D
 encodeDb/StatefulPartitionedCall encodeDb/StatefulPartitionedCall2D
 encodeEa/StatefulPartitionedCall encodeEa/StatefulPartitionedCall2D
 encodeEb/StatefulPartitionedCall encodeEb/StatefulPartitionedCall2V
)spatial_dropout2d/StatefulPartitionedCall)spatial_dropout2d/StatefulPartitionedCall2Z
+spatial_dropout2d_1/StatefulPartitionedCall+spatial_dropout2d_1/StatefulPartitionedCall2H
"transconvA/StatefulPartitionedCall"transconvA/StatefulPartitionedCall2H
"transconvB/StatefulPartitionedCall"transconvB/StatefulPartitionedCall2H
"transconvC/StatefulPartitionedCall"transconvC/StatefulPartitionedCall2H
"transconvE/StatefulPartitionedCall"transconvE/StatefulPartitionedCall:[ W
1
_output_shapes
:�����������
"
_user_specified_name
MRImages
�
j
L__inference_spatial_dropout2d_layer_call_and_return_conditional_losses_18499

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:���������  @2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:���������  @2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:���������  @:W S
/
_output_shapes
:���������  @
 
_user_specified_nameinputs
�

�
I__inference_PredictionMask_layer_call_and_return_conditional_losses_18998

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������*
paddingVALID*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������2	
BiasAddk
SigmoidSigmoidBiasAdd:output:0*
T0*1
_output_shapes
:�����������2	
Sigmoid�
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:�����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�

�
C__inference_encodeBa_layer_call_and_return_conditional_losses_20439

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@ *
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@ 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@@ 2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:���������@@ 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������@@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������@@
 
_user_specified_nameinputs
�
S
'__inference_concatD_layer_call_fn_20753
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_concatD_layer_call_and_return_conditional_losses_186992
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*]
_input_shapesL
J:,����������������������������:����������:l h
B
_output_shapes0
.:,����������������������������
"
_user_specified_name
inputs/0:ZV
0
_output_shapes
:����������
"
_user_specified_name
inputs/1
�

�
C__inference_encodeBb_layer_call_and_return_conditional_losses_20459

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@ *
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@ 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@@ 2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:���������@@ 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������@@ ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������@@ 
 
_user_specified_nameinputs
��
�
Q__inference_2DUNet_Brats_Decathlon_layer_call_and_return_conditional_losses_19967

inputs+
'encodeaa_conv2d_readvariableop_resource,
(encodeaa_biasadd_readvariableop_resource+
'encodeab_conv2d_readvariableop_resource,
(encodeab_biasadd_readvariableop_resource+
'encodeba_conv2d_readvariableop_resource,
(encodeba_biasadd_readvariableop_resource+
'encodebb_conv2d_readvariableop_resource,
(encodebb_biasadd_readvariableop_resource+
'encodeca_conv2d_readvariableop_resource,
(encodeca_biasadd_readvariableop_resource+
'encodecb_conv2d_readvariableop_resource,
(encodecb_biasadd_readvariableop_resource+
'encodeda_conv2d_readvariableop_resource,
(encodeda_biasadd_readvariableop_resource+
'encodedb_conv2d_readvariableop_resource,
(encodedb_biasadd_readvariableop_resource+
'encodeea_conv2d_readvariableop_resource,
(encodeea_biasadd_readvariableop_resource+
'encodeeb_conv2d_readvariableop_resource,
(encodeeb_biasadd_readvariableop_resource7
3transconve_conv2d_transpose_readvariableop_resource.
*transconve_biasadd_readvariableop_resource+
'decodeca_conv2d_readvariableop_resource,
(decodeca_biasadd_readvariableop_resource+
'decodecb_conv2d_readvariableop_resource,
(decodecb_biasadd_readvariableop_resource7
3transconvc_conv2d_transpose_readvariableop_resource.
*transconvc_biasadd_readvariableop_resource+
'decodeba_conv2d_readvariableop_resource,
(decodeba_biasadd_readvariableop_resource+
'decodebb_conv2d_readvariableop_resource,
(decodebb_biasadd_readvariableop_resource7
3transconvb_conv2d_transpose_readvariableop_resource.
*transconvb_biasadd_readvariableop_resource+
'decodeaa_conv2d_readvariableop_resource,
(decodeaa_biasadd_readvariableop_resource+
'decodeab_conv2d_readvariableop_resource,
(decodeab_biasadd_readvariableop_resource7
3transconva_conv2d_transpose_readvariableop_resource.
*transconva_biasadd_readvariableop_resource+
'convouta_conv2d_readvariableop_resource,
(convouta_biasadd_readvariableop_resource+
'convoutb_conv2d_readvariableop_resource,
(convoutb_biasadd_readvariableop_resource1
-predictionmask_conv2d_readvariableop_resource2
.predictionmask_biasadd_readvariableop_resource
identity��%PredictionMask/BiasAdd/ReadVariableOp�$PredictionMask/Conv2D/ReadVariableOp�convOuta/BiasAdd/ReadVariableOp�convOuta/Conv2D/ReadVariableOp�convOutb/BiasAdd/ReadVariableOp�convOutb/Conv2D/ReadVariableOp�decodeAa/BiasAdd/ReadVariableOp�decodeAa/Conv2D/ReadVariableOp�decodeAb/BiasAdd/ReadVariableOp�decodeAb/Conv2D/ReadVariableOp�decodeBa/BiasAdd/ReadVariableOp�decodeBa/Conv2D/ReadVariableOp�decodeBb/BiasAdd/ReadVariableOp�decodeBb/Conv2D/ReadVariableOp�decodeCa/BiasAdd/ReadVariableOp�decodeCa/Conv2D/ReadVariableOp�decodeCb/BiasAdd/ReadVariableOp�decodeCb/Conv2D/ReadVariableOp�encodeAa/BiasAdd/ReadVariableOp�encodeAa/Conv2D/ReadVariableOp�encodeAb/BiasAdd/ReadVariableOp�encodeAb/Conv2D/ReadVariableOp�encodeBa/BiasAdd/ReadVariableOp�encodeBa/Conv2D/ReadVariableOp�encodeBb/BiasAdd/ReadVariableOp�encodeBb/Conv2D/ReadVariableOp�encodeCa/BiasAdd/ReadVariableOp�encodeCa/Conv2D/ReadVariableOp�encodeCb/BiasAdd/ReadVariableOp�encodeCb/Conv2D/ReadVariableOp�encodeDa/BiasAdd/ReadVariableOp�encodeDa/Conv2D/ReadVariableOp�encodeDb/BiasAdd/ReadVariableOp�encodeDb/Conv2D/ReadVariableOp�encodeEa/BiasAdd/ReadVariableOp�encodeEa/Conv2D/ReadVariableOp�encodeEb/BiasAdd/ReadVariableOp�encodeEb/Conv2D/ReadVariableOp�!transconvA/BiasAdd/ReadVariableOp�*transconvA/conv2d_transpose/ReadVariableOp�!transconvB/BiasAdd/ReadVariableOp�*transconvB/conv2d_transpose/ReadVariableOp�!transconvC/BiasAdd/ReadVariableOp�*transconvC/conv2d_transpose/ReadVariableOp�!transconvE/BiasAdd/ReadVariableOp�*transconvE/conv2d_transpose/ReadVariableOp�
encodeAa/Conv2D/ReadVariableOpReadVariableOp'encodeaa_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
encodeAa/Conv2D/ReadVariableOp�
encodeAa/Conv2DConv2Dinputs&encodeAa/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
2
encodeAa/Conv2D�
encodeAa/BiasAdd/ReadVariableOpReadVariableOp(encodeaa_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
encodeAa/BiasAdd/ReadVariableOp�
encodeAa/BiasAddBiasAddencodeAa/Conv2D:output:0'encodeAa/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������2
encodeAa/BiasAdd}
encodeAa/ReluReluencodeAa/BiasAdd:output:0*
T0*1
_output_shapes
:�����������2
encodeAa/Relu�
encodeAb/Conv2D/ReadVariableOpReadVariableOp'encodeab_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
encodeAb/Conv2D/ReadVariableOp�
encodeAb/Conv2DConv2DencodeAa/Relu:activations:0&encodeAb/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
2
encodeAb/Conv2D�
encodeAb/BiasAdd/ReadVariableOpReadVariableOp(encodeab_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
encodeAb/BiasAdd/ReadVariableOp�
encodeAb/BiasAddBiasAddencodeAb/Conv2D:output:0'encodeAb/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������2
encodeAb/BiasAdd}
encodeAb/ReluReluencodeAb/BiasAdd:output:0*
T0*1
_output_shapes
:�����������2
encodeAb/Relu�
poolA/MaxPoolMaxPoolencodeAb/Relu:activations:0*/
_output_shapes
:���������@@*
ksize
*
paddingVALID*
strides
2
poolA/MaxPool�
encodeBa/Conv2D/ReadVariableOpReadVariableOp'encodeba_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
encodeBa/Conv2D/ReadVariableOp�
encodeBa/Conv2DConv2DpoolA/MaxPool:output:0&encodeBa/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@ *
paddingSAME*
strides
2
encodeBa/Conv2D�
encodeBa/BiasAdd/ReadVariableOpReadVariableOp(encodeba_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
encodeBa/BiasAdd/ReadVariableOp�
encodeBa/BiasAddBiasAddencodeBa/Conv2D:output:0'encodeBa/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@ 2
encodeBa/BiasAdd{
encodeBa/ReluReluencodeBa/BiasAdd:output:0*
T0*/
_output_shapes
:���������@@ 2
encodeBa/Relu�
encodeBb/Conv2D/ReadVariableOpReadVariableOp'encodebb_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02 
encodeBb/Conv2D/ReadVariableOp�
encodeBb/Conv2DConv2DencodeBa/Relu:activations:0&encodeBb/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@ *
paddingSAME*
strides
2
encodeBb/Conv2D�
encodeBb/BiasAdd/ReadVariableOpReadVariableOp(encodebb_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
encodeBb/BiasAdd/ReadVariableOp�
encodeBb/BiasAddBiasAddencodeBb/Conv2D:output:0'encodeBb/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@ 2
encodeBb/BiasAdd{
encodeBb/ReluReluencodeBb/BiasAdd:output:0*
T0*/
_output_shapes
:���������@@ 2
encodeBb/Relu�
poolB/MaxPoolMaxPoolencodeBb/Relu:activations:0*/
_output_shapes
:���������   *
ksize
*
paddingVALID*
strides
2
poolB/MaxPool�
encodeCa/Conv2D/ReadVariableOpReadVariableOp'encodeca_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02 
encodeCa/Conv2D/ReadVariableOp�
encodeCa/Conv2DConv2DpoolB/MaxPool:output:0&encodeCa/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  @*
paddingSAME*
strides
2
encodeCa/Conv2D�
encodeCa/BiasAdd/ReadVariableOpReadVariableOp(encodeca_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
encodeCa/BiasAdd/ReadVariableOp�
encodeCa/BiasAddBiasAddencodeCa/Conv2D:output:0'encodeCa/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  @2
encodeCa/BiasAdd{
encodeCa/ReluReluencodeCa/BiasAdd:output:0*
T0*/
_output_shapes
:���������  @2
encodeCa/Relu}
spatial_dropout2d/ShapeShapeencodeCa/Relu:activations:0*
T0*
_output_shapes
:2
spatial_dropout2d/Shape�
%spatial_dropout2d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%spatial_dropout2d/strided_slice/stack�
'spatial_dropout2d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'spatial_dropout2d/strided_slice/stack_1�
'spatial_dropout2d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'spatial_dropout2d/strided_slice/stack_2�
spatial_dropout2d/strided_sliceStridedSlice spatial_dropout2d/Shape:output:0.spatial_dropout2d/strided_slice/stack:output:00spatial_dropout2d/strided_slice/stack_1:output:00spatial_dropout2d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
spatial_dropout2d/strided_slice�
'spatial_dropout2d/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2)
'spatial_dropout2d/strided_slice_1/stack�
)spatial_dropout2d/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)spatial_dropout2d/strided_slice_1/stack_1�
)spatial_dropout2d/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)spatial_dropout2d/strided_slice_1/stack_2�
!spatial_dropout2d/strided_slice_1StridedSlice spatial_dropout2d/Shape:output:00spatial_dropout2d/strided_slice_1/stack:output:02spatial_dropout2d/strided_slice_1/stack_1:output:02spatial_dropout2d/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!spatial_dropout2d/strided_slice_1�
spatial_dropout2d/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2!
spatial_dropout2d/dropout/Const�
spatial_dropout2d/dropout/MulMulencodeCa/Relu:activations:0(spatial_dropout2d/dropout/Const:output:0*
T0*/
_output_shapes
:���������  @2
spatial_dropout2d/dropout/Mul�
0spatial_dropout2d/dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :22
0spatial_dropout2d/dropout/random_uniform/shape/1�
0spatial_dropout2d/dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :22
0spatial_dropout2d/dropout/random_uniform/shape/2�
.spatial_dropout2d/dropout/random_uniform/shapePack(spatial_dropout2d/strided_slice:output:09spatial_dropout2d/dropout/random_uniform/shape/1:output:09spatial_dropout2d/dropout/random_uniform/shape/2:output:0*spatial_dropout2d/strided_slice_1:output:0*
N*
T0*
_output_shapes
:20
.spatial_dropout2d/dropout/random_uniform/shape�
6spatial_dropout2d/dropout/random_uniform/RandomUniformRandomUniform7spatial_dropout2d/dropout/random_uniform/shape:output:0*
T0*8
_output_shapes&
$:"������������������*
dtype028
6spatial_dropout2d/dropout/random_uniform/RandomUniform�
(spatial_dropout2d/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2*
(spatial_dropout2d/dropout/GreaterEqual/y�
&spatial_dropout2d/dropout/GreaterEqualGreaterEqual?spatial_dropout2d/dropout/random_uniform/RandomUniform:output:01spatial_dropout2d/dropout/GreaterEqual/y:output:0*
T0*8
_output_shapes&
$:"������������������2(
&spatial_dropout2d/dropout/GreaterEqual�
spatial_dropout2d/dropout/CastCast*spatial_dropout2d/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*8
_output_shapes&
$:"������������������2 
spatial_dropout2d/dropout/Cast�
spatial_dropout2d/dropout/Mul_1Mul!spatial_dropout2d/dropout/Mul:z:0"spatial_dropout2d/dropout/Cast:y:0*
T0*/
_output_shapes
:���������  @2!
spatial_dropout2d/dropout/Mul_1�
encodeCb/Conv2D/ReadVariableOpReadVariableOp'encodecb_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
encodeCb/Conv2D/ReadVariableOp�
encodeCb/Conv2DConv2D#spatial_dropout2d/dropout/Mul_1:z:0&encodeCb/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  @*
paddingSAME*
strides
2
encodeCb/Conv2D�
encodeCb/BiasAdd/ReadVariableOpReadVariableOp(encodecb_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
encodeCb/BiasAdd/ReadVariableOp�
encodeCb/BiasAddBiasAddencodeCb/Conv2D:output:0'encodeCb/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  @2
encodeCb/BiasAdd{
encodeCb/ReluReluencodeCb/BiasAdd:output:0*
T0*/
_output_shapes
:���������  @2
encodeCb/Relu�
poolC/MaxPoolMaxPoolencodeCb/Relu:activations:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
2
poolC/MaxPool�
encodeDa/Conv2D/ReadVariableOpReadVariableOp'encodeda_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype02 
encodeDa/Conv2D/ReadVariableOp�
encodeDa/Conv2DConv2DpoolC/MaxPool:output:0&encodeDa/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
encodeDa/Conv2D�
encodeDa/BiasAdd/ReadVariableOpReadVariableOp(encodeda_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
encodeDa/BiasAdd/ReadVariableOp�
encodeDa/BiasAddBiasAddencodeDa/Conv2D:output:0'encodeDa/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
encodeDa/BiasAdd|
encodeDa/ReluReluencodeDa/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
encodeDa/Relu�
spatial_dropout2d_1/ShapeShapeencodeDa/Relu:activations:0*
T0*
_output_shapes
:2
spatial_dropout2d_1/Shape�
'spatial_dropout2d_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'spatial_dropout2d_1/strided_slice/stack�
)spatial_dropout2d_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)spatial_dropout2d_1/strided_slice/stack_1�
)spatial_dropout2d_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)spatial_dropout2d_1/strided_slice/stack_2�
!spatial_dropout2d_1/strided_sliceStridedSlice"spatial_dropout2d_1/Shape:output:00spatial_dropout2d_1/strided_slice/stack:output:02spatial_dropout2d_1/strided_slice/stack_1:output:02spatial_dropout2d_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!spatial_dropout2d_1/strided_slice�
)spatial_dropout2d_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2+
)spatial_dropout2d_1/strided_slice_1/stack�
+spatial_dropout2d_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+spatial_dropout2d_1/strided_slice_1/stack_1�
+spatial_dropout2d_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+spatial_dropout2d_1/strided_slice_1/stack_2�
#spatial_dropout2d_1/strided_slice_1StridedSlice"spatial_dropout2d_1/Shape:output:02spatial_dropout2d_1/strided_slice_1/stack:output:04spatial_dropout2d_1/strided_slice_1/stack_1:output:04spatial_dropout2d_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#spatial_dropout2d_1/strided_slice_1�
!spatial_dropout2d_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2#
!spatial_dropout2d_1/dropout/Const�
spatial_dropout2d_1/dropout/MulMulencodeDa/Relu:activations:0*spatial_dropout2d_1/dropout/Const:output:0*
T0*0
_output_shapes
:����������2!
spatial_dropout2d_1/dropout/Mul�
2spatial_dropout2d_1/dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :24
2spatial_dropout2d_1/dropout/random_uniform/shape/1�
2spatial_dropout2d_1/dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :24
2spatial_dropout2d_1/dropout/random_uniform/shape/2�
0spatial_dropout2d_1/dropout/random_uniform/shapePack*spatial_dropout2d_1/strided_slice:output:0;spatial_dropout2d_1/dropout/random_uniform/shape/1:output:0;spatial_dropout2d_1/dropout/random_uniform/shape/2:output:0,spatial_dropout2d_1/strided_slice_1:output:0*
N*
T0*
_output_shapes
:22
0spatial_dropout2d_1/dropout/random_uniform/shape�
8spatial_dropout2d_1/dropout/random_uniform/RandomUniformRandomUniform9spatial_dropout2d_1/dropout/random_uniform/shape:output:0*
T0*8
_output_shapes&
$:"������������������*
dtype02:
8spatial_dropout2d_1/dropout/random_uniform/RandomUniform�
*spatial_dropout2d_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2,
*spatial_dropout2d_1/dropout/GreaterEqual/y�
(spatial_dropout2d_1/dropout/GreaterEqualGreaterEqualAspatial_dropout2d_1/dropout/random_uniform/RandomUniform:output:03spatial_dropout2d_1/dropout/GreaterEqual/y:output:0*
T0*8
_output_shapes&
$:"������������������2*
(spatial_dropout2d_1/dropout/GreaterEqual�
 spatial_dropout2d_1/dropout/CastCast,spatial_dropout2d_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*8
_output_shapes&
$:"������������������2"
 spatial_dropout2d_1/dropout/Cast�
!spatial_dropout2d_1/dropout/Mul_1Mul#spatial_dropout2d_1/dropout/Mul:z:0$spatial_dropout2d_1/dropout/Cast:y:0*
T0*0
_output_shapes
:����������2#
!spatial_dropout2d_1/dropout/Mul_1�
encodeDb/Conv2D/ReadVariableOpReadVariableOp'encodedb_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02 
encodeDb/Conv2D/ReadVariableOp�
encodeDb/Conv2DConv2D%spatial_dropout2d_1/dropout/Mul_1:z:0&encodeDb/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
encodeDb/Conv2D�
encodeDb/BiasAdd/ReadVariableOpReadVariableOp(encodedb_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
encodeDb/BiasAdd/ReadVariableOp�
encodeDb/BiasAddBiasAddencodeDb/Conv2D:output:0'encodeDb/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
encodeDb/BiasAdd|
encodeDb/ReluReluencodeDb/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
encodeDb/Relu�
poolD/MaxPoolMaxPoolencodeDb/Relu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2
poolD/MaxPool�
encodeEa/Conv2D/ReadVariableOpReadVariableOp'encodeea_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02 
encodeEa/Conv2D/ReadVariableOp�
encodeEa/Conv2DConv2DpoolD/MaxPool:output:0&encodeEa/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
encodeEa/Conv2D�
encodeEa/BiasAdd/ReadVariableOpReadVariableOp(encodeea_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
encodeEa/BiasAdd/ReadVariableOp�
encodeEa/BiasAddBiasAddencodeEa/Conv2D:output:0'encodeEa/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
encodeEa/BiasAdd|
encodeEa/ReluReluencodeEa/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
encodeEa/Relu�
encodeEb/Conv2D/ReadVariableOpReadVariableOp'encodeeb_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02 
encodeEb/Conv2D/ReadVariableOp�
encodeEb/Conv2DConv2DencodeEa/Relu:activations:0&encodeEb/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
encodeEb/Conv2D�
encodeEb/BiasAdd/ReadVariableOpReadVariableOp(encodeeb_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
encodeEb/BiasAdd/ReadVariableOp�
encodeEb/BiasAddBiasAddencodeEb/Conv2D:output:0'encodeEb/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
encodeEb/BiasAdd|
encodeEb/ReluReluencodeEb/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
encodeEb/Reluo
transconvE/ShapeShapeencodeEb/Relu:activations:0*
T0*
_output_shapes
:2
transconvE/Shape�
transconvE/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
transconvE/strided_slice/stack�
 transconvE/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 transconvE/strided_slice/stack_1�
 transconvE/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 transconvE/strided_slice/stack_2�
transconvE/strided_sliceStridedSlicetransconvE/Shape:output:0'transconvE/strided_slice/stack:output:0)transconvE/strided_slice/stack_1:output:0)transconvE/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
transconvE/strided_slicej
transconvE/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
transconvE/stack/1j
transconvE/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
transconvE/stack/2k
transconvE/stack/3Const*
_output_shapes
: *
dtype0*
value
B :�2
transconvE/stack/3�
transconvE/stackPack!transconvE/strided_slice:output:0transconvE/stack/1:output:0transconvE/stack/2:output:0transconvE/stack/3:output:0*
N*
T0*
_output_shapes
:2
transconvE/stack�
 transconvE/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 transconvE/strided_slice_1/stack�
"transconvE/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"transconvE/strided_slice_1/stack_1�
"transconvE/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"transconvE/strided_slice_1/stack_2�
transconvE/strided_slice_1StridedSlicetransconvE/stack:output:0)transconvE/strided_slice_1/stack:output:0+transconvE/strided_slice_1/stack_1:output:0+transconvE/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
transconvE/strided_slice_1�
*transconvE/conv2d_transpose/ReadVariableOpReadVariableOp3transconve_conv2d_transpose_readvariableop_resource*(
_output_shapes
:��*
dtype02,
*transconvE/conv2d_transpose/ReadVariableOp�
transconvE/conv2d_transposeConv2DBackpropInputtransconvE/stack:output:02transconvE/conv2d_transpose/ReadVariableOp:value:0encodeEb/Relu:activations:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
transconvE/conv2d_transpose�
!transconvE/BiasAdd/ReadVariableOpReadVariableOp*transconve_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02#
!transconvE/BiasAdd/ReadVariableOp�
transconvE/BiasAddBiasAdd$transconvE/conv2d_transpose:output:0)transconvE/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
transconvE/BiasAddl
concatD/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatD/concat/axis�
concatD/concatConcatV2transconvE/BiasAdd:output:0encodeDb/Relu:activations:0concatD/concat/axis:output:0*
N*
T0*0
_output_shapes
:����������2
concatD/concat�
decodeCa/Conv2D/ReadVariableOpReadVariableOp'decodeca_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02 
decodeCa/Conv2D/ReadVariableOp�
decodeCa/Conv2DConv2DconcatD/concat:output:0&decodeCa/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
decodeCa/Conv2D�
decodeCa/BiasAdd/ReadVariableOpReadVariableOp(decodeca_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
decodeCa/BiasAdd/ReadVariableOp�
decodeCa/BiasAddBiasAdddecodeCa/Conv2D:output:0'decodeCa/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
decodeCa/BiasAdd|
decodeCa/ReluReludecodeCa/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
decodeCa/Relu�
decodeCb/Conv2D/ReadVariableOpReadVariableOp'decodecb_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02 
decodeCb/Conv2D/ReadVariableOp�
decodeCb/Conv2DConv2DdecodeCa/Relu:activations:0&decodeCb/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
decodeCb/Conv2D�
decodeCb/BiasAdd/ReadVariableOpReadVariableOp(decodecb_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
decodeCb/BiasAdd/ReadVariableOp�
decodeCb/BiasAddBiasAdddecodeCb/Conv2D:output:0'decodeCb/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
decodeCb/BiasAdd|
decodeCb/ReluReludecodeCb/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
decodeCb/Reluo
transconvC/ShapeShapedecodeCb/Relu:activations:0*
T0*
_output_shapes
:2
transconvC/Shape�
transconvC/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
transconvC/strided_slice/stack�
 transconvC/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 transconvC/strided_slice/stack_1�
 transconvC/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 transconvC/strided_slice/stack_2�
transconvC/strided_sliceStridedSlicetransconvC/Shape:output:0'transconvC/strided_slice/stack:output:0)transconvC/strided_slice/stack_1:output:0)transconvC/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
transconvC/strided_slicej
transconvC/stack/1Const*
_output_shapes
: *
dtype0*
value	B : 2
transconvC/stack/1j
transconvC/stack/2Const*
_output_shapes
: *
dtype0*
value	B : 2
transconvC/stack/2j
transconvC/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2
transconvC/stack/3�
transconvC/stackPack!transconvC/strided_slice:output:0transconvC/stack/1:output:0transconvC/stack/2:output:0transconvC/stack/3:output:0*
N*
T0*
_output_shapes
:2
transconvC/stack�
 transconvC/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 transconvC/strided_slice_1/stack�
"transconvC/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"transconvC/strided_slice_1/stack_1�
"transconvC/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"transconvC/strided_slice_1/stack_2�
transconvC/strided_slice_1StridedSlicetransconvC/stack:output:0)transconvC/strided_slice_1/stack:output:0+transconvC/strided_slice_1/stack_1:output:0+transconvC/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
transconvC/strided_slice_1�
*transconvC/conv2d_transpose/ReadVariableOpReadVariableOp3transconvc_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@�*
dtype02,
*transconvC/conv2d_transpose/ReadVariableOp�
transconvC/conv2d_transposeConv2DBackpropInputtransconvC/stack:output:02transconvC/conv2d_transpose/ReadVariableOp:value:0decodeCb/Relu:activations:0*
T0*/
_output_shapes
:���������  @*
paddingSAME*
strides
2
transconvC/conv2d_transpose�
!transconvC/BiasAdd/ReadVariableOpReadVariableOp*transconvc_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02#
!transconvC/BiasAdd/ReadVariableOp�
transconvC/BiasAddBiasAdd$transconvC/conv2d_transpose:output:0)transconvC/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  @2
transconvC/BiasAddl
concatC/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatC/concat/axis�
concatC/concatConcatV2transconvC/BiasAdd:output:0encodeCb/Relu:activations:0concatC/concat/axis:output:0*
N*
T0*0
_output_shapes
:���������  �2
concatC/concat�
decodeBa/Conv2D/ReadVariableOpReadVariableOp'decodeba_conv2d_readvariableop_resource*'
_output_shapes
:�@*
dtype02 
decodeBa/Conv2D/ReadVariableOp�
decodeBa/Conv2DConv2DconcatC/concat:output:0&decodeBa/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  @*
paddingSAME*
strides
2
decodeBa/Conv2D�
decodeBa/BiasAdd/ReadVariableOpReadVariableOp(decodeba_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
decodeBa/BiasAdd/ReadVariableOp�
decodeBa/BiasAddBiasAdddecodeBa/Conv2D:output:0'decodeBa/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  @2
decodeBa/BiasAdd{
decodeBa/ReluReludecodeBa/BiasAdd:output:0*
T0*/
_output_shapes
:���������  @2
decodeBa/Relu�
decodeBb/Conv2D/ReadVariableOpReadVariableOp'decodebb_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
decodeBb/Conv2D/ReadVariableOp�
decodeBb/Conv2DConv2DdecodeBa/Relu:activations:0&decodeBb/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  @*
paddingSAME*
strides
2
decodeBb/Conv2D�
decodeBb/BiasAdd/ReadVariableOpReadVariableOp(decodebb_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
decodeBb/BiasAdd/ReadVariableOp�
decodeBb/BiasAddBiasAdddecodeBb/Conv2D:output:0'decodeBb/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  @2
decodeBb/BiasAdd{
decodeBb/ReluReludecodeBb/BiasAdd:output:0*
T0*/
_output_shapes
:���������  @2
decodeBb/Reluo
transconvB/ShapeShapedecodeBb/Relu:activations:0*
T0*
_output_shapes
:2
transconvB/Shape�
transconvB/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
transconvB/strided_slice/stack�
 transconvB/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 transconvB/strided_slice/stack_1�
 transconvB/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 transconvB/strided_slice/stack_2�
transconvB/strided_sliceStridedSlicetransconvB/Shape:output:0'transconvB/strided_slice/stack:output:0)transconvB/strided_slice/stack_1:output:0)transconvB/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
transconvB/strided_slicej
transconvB/stack/1Const*
_output_shapes
: *
dtype0*
value	B :@2
transconvB/stack/1j
transconvB/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@2
transconvB/stack/2j
transconvB/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2
transconvB/stack/3�
transconvB/stackPack!transconvB/strided_slice:output:0transconvB/stack/1:output:0transconvB/stack/2:output:0transconvB/stack/3:output:0*
N*
T0*
_output_shapes
:2
transconvB/stack�
 transconvB/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 transconvB/strided_slice_1/stack�
"transconvB/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"transconvB/strided_slice_1/stack_1�
"transconvB/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"transconvB/strided_slice_1/stack_2�
transconvB/strided_slice_1StridedSlicetransconvB/stack:output:0)transconvB/strided_slice_1/stack:output:0+transconvB/strided_slice_1/stack_1:output:0+transconvB/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
transconvB/strided_slice_1�
*transconvB/conv2d_transpose/ReadVariableOpReadVariableOp3transconvb_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype02,
*transconvB/conv2d_transpose/ReadVariableOp�
transconvB/conv2d_transposeConv2DBackpropInputtransconvB/stack:output:02transconvB/conv2d_transpose/ReadVariableOp:value:0decodeBb/Relu:activations:0*
T0*/
_output_shapes
:���������@@ *
paddingSAME*
strides
2
transconvB/conv2d_transpose�
!transconvB/BiasAdd/ReadVariableOpReadVariableOp*transconvb_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02#
!transconvB/BiasAdd/ReadVariableOp�
transconvB/BiasAddBiasAdd$transconvB/conv2d_transpose:output:0)transconvB/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@ 2
transconvB/BiasAddl
concatB/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatB/concat/axis�
concatB/concatConcatV2transconvB/BiasAdd:output:0encodeBb/Relu:activations:0concatB/concat/axis:output:0*
N*
T0*/
_output_shapes
:���������@@@2
concatB/concat�
decodeAa/Conv2D/ReadVariableOpReadVariableOp'decodeaa_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02 
decodeAa/Conv2D/ReadVariableOp�
decodeAa/Conv2DConv2DconcatB/concat:output:0&decodeAa/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@ *
paddingSAME*
strides
2
decodeAa/Conv2D�
decodeAa/BiasAdd/ReadVariableOpReadVariableOp(decodeaa_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
decodeAa/BiasAdd/ReadVariableOp�
decodeAa/BiasAddBiasAdddecodeAa/Conv2D:output:0'decodeAa/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@ 2
decodeAa/BiasAdd{
decodeAa/ReluReludecodeAa/BiasAdd:output:0*
T0*/
_output_shapes
:���������@@ 2
decodeAa/Relu�
decodeAb/Conv2D/ReadVariableOpReadVariableOp'decodeab_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02 
decodeAb/Conv2D/ReadVariableOp�
decodeAb/Conv2DConv2DdecodeAa/Relu:activations:0&decodeAb/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@ *
paddingSAME*
strides
2
decodeAb/Conv2D�
decodeAb/BiasAdd/ReadVariableOpReadVariableOp(decodeab_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
decodeAb/BiasAdd/ReadVariableOp�
decodeAb/BiasAddBiasAdddecodeAb/Conv2D:output:0'decodeAb/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@ 2
decodeAb/BiasAdd{
decodeAb/ReluReludecodeAb/BiasAdd:output:0*
T0*/
_output_shapes
:���������@@ 2
decodeAb/Reluo
transconvA/ShapeShapedecodeAb/Relu:activations:0*
T0*
_output_shapes
:2
transconvA/Shape�
transconvA/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
transconvA/strided_slice/stack�
 transconvA/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 transconvA/strided_slice/stack_1�
 transconvA/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 transconvA/strided_slice/stack_2�
transconvA/strided_sliceStridedSlicetransconvA/Shape:output:0'transconvA/strided_slice/stack:output:0)transconvA/strided_slice/stack_1:output:0)transconvA/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
transconvA/strided_slicek
transconvA/stack/1Const*
_output_shapes
: *
dtype0*
value
B :�2
transconvA/stack/1k
transconvA/stack/2Const*
_output_shapes
: *
dtype0*
value
B :�2
transconvA/stack/2j
transconvA/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
transconvA/stack/3�
transconvA/stackPack!transconvA/strided_slice:output:0transconvA/stack/1:output:0transconvA/stack/2:output:0transconvA/stack/3:output:0*
N*
T0*
_output_shapes
:2
transconvA/stack�
 transconvA/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 transconvA/strided_slice_1/stack�
"transconvA/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"transconvA/strided_slice_1/stack_1�
"transconvA/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"transconvA/strided_slice_1/stack_2�
transconvA/strided_slice_1StridedSlicetransconvA/stack:output:0)transconvA/strided_slice_1/stack:output:0+transconvA/strided_slice_1/stack_1:output:0+transconvA/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
transconvA/strided_slice_1�
*transconvA/conv2d_transpose/ReadVariableOpReadVariableOp3transconva_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02,
*transconvA/conv2d_transpose/ReadVariableOp�
transconvA/conv2d_transposeConv2DBackpropInputtransconvA/stack:output:02transconvA/conv2d_transpose/ReadVariableOp:value:0decodeAb/Relu:activations:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
2
transconvA/conv2d_transpose�
!transconvA/BiasAdd/ReadVariableOpReadVariableOp*transconva_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!transconvA/BiasAdd/ReadVariableOp�
transconvA/BiasAddBiasAdd$transconvA/conv2d_transpose:output:0)transconvA/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������2
transconvA/BiasAddl
concatA/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatA/concat/axis�
concatA/concatConcatV2transconvA/BiasAdd:output:0encodeAb/Relu:activations:0concatA/concat/axis:output:0*
N*
T0*1
_output_shapes
:����������� 2
concatA/concat�
convOuta/Conv2D/ReadVariableOpReadVariableOp'convouta_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
convOuta/Conv2D/ReadVariableOp�
convOuta/Conv2DConv2DconcatA/concat:output:0&convOuta/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
2
convOuta/Conv2D�
convOuta/BiasAdd/ReadVariableOpReadVariableOp(convouta_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
convOuta/BiasAdd/ReadVariableOp�
convOuta/BiasAddBiasAddconvOuta/Conv2D:output:0'convOuta/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������2
convOuta/BiasAdd}
convOuta/ReluReluconvOuta/BiasAdd:output:0*
T0*1
_output_shapes
:�����������2
convOuta/Relu�
convOutb/Conv2D/ReadVariableOpReadVariableOp'convoutb_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
convOutb/Conv2D/ReadVariableOp�
convOutb/Conv2DConv2DconvOuta/Relu:activations:0&convOutb/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
2
convOutb/Conv2D�
convOutb/BiasAdd/ReadVariableOpReadVariableOp(convoutb_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
convOutb/BiasAdd/ReadVariableOp�
convOutb/BiasAddBiasAddconvOutb/Conv2D:output:0'convOutb/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������2
convOutb/BiasAdd}
convOutb/ReluReluconvOutb/BiasAdd:output:0*
T0*1
_output_shapes
:�����������2
convOutb/Relu�
$PredictionMask/Conv2D/ReadVariableOpReadVariableOp-predictionmask_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02&
$PredictionMask/Conv2D/ReadVariableOp�
PredictionMask/Conv2DConv2DconvOutb/Relu:activations:0,PredictionMask/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������*
paddingVALID*
strides
2
PredictionMask/Conv2D�
%PredictionMask/BiasAdd/ReadVariableOpReadVariableOp.predictionmask_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%PredictionMask/BiasAdd/ReadVariableOp�
PredictionMask/BiasAddBiasAddPredictionMask/Conv2D:output:0-PredictionMask/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������2
PredictionMask/BiasAdd�
PredictionMask/SigmoidSigmoidPredictionMask/BiasAdd:output:0*
T0*1
_output_shapes
:�����������2
PredictionMask/Sigmoid�
IdentityIdentityPredictionMask/Sigmoid:y:0&^PredictionMask/BiasAdd/ReadVariableOp%^PredictionMask/Conv2D/ReadVariableOp ^convOuta/BiasAdd/ReadVariableOp^convOuta/Conv2D/ReadVariableOp ^convOutb/BiasAdd/ReadVariableOp^convOutb/Conv2D/ReadVariableOp ^decodeAa/BiasAdd/ReadVariableOp^decodeAa/Conv2D/ReadVariableOp ^decodeAb/BiasAdd/ReadVariableOp^decodeAb/Conv2D/ReadVariableOp ^decodeBa/BiasAdd/ReadVariableOp^decodeBa/Conv2D/ReadVariableOp ^decodeBb/BiasAdd/ReadVariableOp^decodeBb/Conv2D/ReadVariableOp ^decodeCa/BiasAdd/ReadVariableOp^decodeCa/Conv2D/ReadVariableOp ^decodeCb/BiasAdd/ReadVariableOp^decodeCb/Conv2D/ReadVariableOp ^encodeAa/BiasAdd/ReadVariableOp^encodeAa/Conv2D/ReadVariableOp ^encodeAb/BiasAdd/ReadVariableOp^encodeAb/Conv2D/ReadVariableOp ^encodeBa/BiasAdd/ReadVariableOp^encodeBa/Conv2D/ReadVariableOp ^encodeBb/BiasAdd/ReadVariableOp^encodeBb/Conv2D/ReadVariableOp ^encodeCa/BiasAdd/ReadVariableOp^encodeCa/Conv2D/ReadVariableOp ^encodeCb/BiasAdd/ReadVariableOp^encodeCb/Conv2D/ReadVariableOp ^encodeDa/BiasAdd/ReadVariableOp^encodeDa/Conv2D/ReadVariableOp ^encodeDb/BiasAdd/ReadVariableOp^encodeDb/Conv2D/ReadVariableOp ^encodeEa/BiasAdd/ReadVariableOp^encodeEa/Conv2D/ReadVariableOp ^encodeEb/BiasAdd/ReadVariableOp^encodeEb/Conv2D/ReadVariableOp"^transconvA/BiasAdd/ReadVariableOp+^transconvA/conv2d_transpose/ReadVariableOp"^transconvB/BiasAdd/ReadVariableOp+^transconvB/conv2d_transpose/ReadVariableOp"^transconvC/BiasAdd/ReadVariableOp+^transconvC/conv2d_transpose/ReadVariableOp"^transconvE/BiasAdd/ReadVariableOp+^transconvE/conv2d_transpose/ReadVariableOp*
T0*1
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:�����������::::::::::::::::::::::::::::::::::::::::::::::2N
%PredictionMask/BiasAdd/ReadVariableOp%PredictionMask/BiasAdd/ReadVariableOp2L
$PredictionMask/Conv2D/ReadVariableOp$PredictionMask/Conv2D/ReadVariableOp2B
convOuta/BiasAdd/ReadVariableOpconvOuta/BiasAdd/ReadVariableOp2@
convOuta/Conv2D/ReadVariableOpconvOuta/Conv2D/ReadVariableOp2B
convOutb/BiasAdd/ReadVariableOpconvOutb/BiasAdd/ReadVariableOp2@
convOutb/Conv2D/ReadVariableOpconvOutb/Conv2D/ReadVariableOp2B
decodeAa/BiasAdd/ReadVariableOpdecodeAa/BiasAdd/ReadVariableOp2@
decodeAa/Conv2D/ReadVariableOpdecodeAa/Conv2D/ReadVariableOp2B
decodeAb/BiasAdd/ReadVariableOpdecodeAb/BiasAdd/ReadVariableOp2@
decodeAb/Conv2D/ReadVariableOpdecodeAb/Conv2D/ReadVariableOp2B
decodeBa/BiasAdd/ReadVariableOpdecodeBa/BiasAdd/ReadVariableOp2@
decodeBa/Conv2D/ReadVariableOpdecodeBa/Conv2D/ReadVariableOp2B
decodeBb/BiasAdd/ReadVariableOpdecodeBb/BiasAdd/ReadVariableOp2@
decodeBb/Conv2D/ReadVariableOpdecodeBb/Conv2D/ReadVariableOp2B
decodeCa/BiasAdd/ReadVariableOpdecodeCa/BiasAdd/ReadVariableOp2@
decodeCa/Conv2D/ReadVariableOpdecodeCa/Conv2D/ReadVariableOp2B
decodeCb/BiasAdd/ReadVariableOpdecodeCb/BiasAdd/ReadVariableOp2@
decodeCb/Conv2D/ReadVariableOpdecodeCb/Conv2D/ReadVariableOp2B
encodeAa/BiasAdd/ReadVariableOpencodeAa/BiasAdd/ReadVariableOp2@
encodeAa/Conv2D/ReadVariableOpencodeAa/Conv2D/ReadVariableOp2B
encodeAb/BiasAdd/ReadVariableOpencodeAb/BiasAdd/ReadVariableOp2@
encodeAb/Conv2D/ReadVariableOpencodeAb/Conv2D/ReadVariableOp2B
encodeBa/BiasAdd/ReadVariableOpencodeBa/BiasAdd/ReadVariableOp2@
encodeBa/Conv2D/ReadVariableOpencodeBa/Conv2D/ReadVariableOp2B
encodeBb/BiasAdd/ReadVariableOpencodeBb/BiasAdd/ReadVariableOp2@
encodeBb/Conv2D/ReadVariableOpencodeBb/Conv2D/ReadVariableOp2B
encodeCa/BiasAdd/ReadVariableOpencodeCa/BiasAdd/ReadVariableOp2@
encodeCa/Conv2D/ReadVariableOpencodeCa/Conv2D/ReadVariableOp2B
encodeCb/BiasAdd/ReadVariableOpencodeCb/BiasAdd/ReadVariableOp2@
encodeCb/Conv2D/ReadVariableOpencodeCb/Conv2D/ReadVariableOp2B
encodeDa/BiasAdd/ReadVariableOpencodeDa/BiasAdd/ReadVariableOp2@
encodeDa/Conv2D/ReadVariableOpencodeDa/Conv2D/ReadVariableOp2B
encodeDb/BiasAdd/ReadVariableOpencodeDb/BiasAdd/ReadVariableOp2@
encodeDb/Conv2D/ReadVariableOpencodeDb/Conv2D/ReadVariableOp2B
encodeEa/BiasAdd/ReadVariableOpencodeEa/BiasAdd/ReadVariableOp2@
encodeEa/Conv2D/ReadVariableOpencodeEa/Conv2D/ReadVariableOp2B
encodeEb/BiasAdd/ReadVariableOpencodeEb/BiasAdd/ReadVariableOp2@
encodeEb/Conv2D/ReadVariableOpencodeEb/Conv2D/ReadVariableOp2F
!transconvA/BiasAdd/ReadVariableOp!transconvA/BiasAdd/ReadVariableOp2X
*transconvA/conv2d_transpose/ReadVariableOp*transconvA/conv2d_transpose/ReadVariableOp2F
!transconvB/BiasAdd/ReadVariableOp!transconvB/BiasAdd/ReadVariableOp2X
*transconvB/conv2d_transpose/ReadVariableOp*transconvB/conv2d_transpose/ReadVariableOp2F
!transconvC/BiasAdd/ReadVariableOp!transconvC/BiasAdd/ReadVariableOp2X
*transconvC/conv2d_transpose/ReadVariableOp*transconvC/conv2d_transpose/ReadVariableOp2F
!transconvE/BiasAdd/ReadVariableOp!transconvE/BiasAdd/ReadVariableOp2X
*transconvE/conv2d_transpose/ReadVariableOp*transconvE/conv2d_transpose/ReadVariableOp:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
}
(__inference_decodeCb_layer_call_fn_20793

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_decodeCb_layer_call_and_return_conditional_losses_187462
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
C__inference_decodeBb_layer_call_and_return_conditional_losses_18821

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  @*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  @2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������  @2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:���������  @2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������  @::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������  @
 
_user_specified_nameinputs
�
j
1__inference_spatial_dropout2d_layer_call_fn_20521

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_spatial_dropout2d_layer_call_and_return_conditional_losses_180502
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������22
StatefulPartitionedCallStatefulPartitionedCall:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
l
3__inference_spatial_dropout2d_1_layer_call_fn_20637

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_spatial_dropout2d_1_layer_call_and_return_conditional_losses_185882
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
C__inference_decodeCb_layer_call_and_return_conditional_losses_20784

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:����������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
C__inference_encodeDb_layer_call_and_return_conditional_losses_18616

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:����������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
C__inference_encodeBb_layer_call_and_return_conditional_losses_18428

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@ *
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@ 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@@ 2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:���������@@ 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������@@ ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������@@ 
 
_user_specified_nameinputs
�
j
1__inference_spatial_dropout2d_layer_call_fn_20559

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_spatial_dropout2d_layer_call_and_return_conditional_losses_184942
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������  @2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������  @22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������  @
 
_user_specified_nameinputs
�
�
Q__inference_2DUNet_Brats_Decathlon_layer_call_and_return_conditional_losses_20194

inputs+
'encodeaa_conv2d_readvariableop_resource,
(encodeaa_biasadd_readvariableop_resource+
'encodeab_conv2d_readvariableop_resource,
(encodeab_biasadd_readvariableop_resource+
'encodeba_conv2d_readvariableop_resource,
(encodeba_biasadd_readvariableop_resource+
'encodebb_conv2d_readvariableop_resource,
(encodebb_biasadd_readvariableop_resource+
'encodeca_conv2d_readvariableop_resource,
(encodeca_biasadd_readvariableop_resource+
'encodecb_conv2d_readvariableop_resource,
(encodecb_biasadd_readvariableop_resource+
'encodeda_conv2d_readvariableop_resource,
(encodeda_biasadd_readvariableop_resource+
'encodedb_conv2d_readvariableop_resource,
(encodedb_biasadd_readvariableop_resource+
'encodeea_conv2d_readvariableop_resource,
(encodeea_biasadd_readvariableop_resource+
'encodeeb_conv2d_readvariableop_resource,
(encodeeb_biasadd_readvariableop_resource7
3transconve_conv2d_transpose_readvariableop_resource.
*transconve_biasadd_readvariableop_resource+
'decodeca_conv2d_readvariableop_resource,
(decodeca_biasadd_readvariableop_resource+
'decodecb_conv2d_readvariableop_resource,
(decodecb_biasadd_readvariableop_resource7
3transconvc_conv2d_transpose_readvariableop_resource.
*transconvc_biasadd_readvariableop_resource+
'decodeba_conv2d_readvariableop_resource,
(decodeba_biasadd_readvariableop_resource+
'decodebb_conv2d_readvariableop_resource,
(decodebb_biasadd_readvariableop_resource7
3transconvb_conv2d_transpose_readvariableop_resource.
*transconvb_biasadd_readvariableop_resource+
'decodeaa_conv2d_readvariableop_resource,
(decodeaa_biasadd_readvariableop_resource+
'decodeab_conv2d_readvariableop_resource,
(decodeab_biasadd_readvariableop_resource7
3transconva_conv2d_transpose_readvariableop_resource.
*transconva_biasadd_readvariableop_resource+
'convouta_conv2d_readvariableop_resource,
(convouta_biasadd_readvariableop_resource+
'convoutb_conv2d_readvariableop_resource,
(convoutb_biasadd_readvariableop_resource1
-predictionmask_conv2d_readvariableop_resource2
.predictionmask_biasadd_readvariableop_resource
identity��%PredictionMask/BiasAdd/ReadVariableOp�$PredictionMask/Conv2D/ReadVariableOp�convOuta/BiasAdd/ReadVariableOp�convOuta/Conv2D/ReadVariableOp�convOutb/BiasAdd/ReadVariableOp�convOutb/Conv2D/ReadVariableOp�decodeAa/BiasAdd/ReadVariableOp�decodeAa/Conv2D/ReadVariableOp�decodeAb/BiasAdd/ReadVariableOp�decodeAb/Conv2D/ReadVariableOp�decodeBa/BiasAdd/ReadVariableOp�decodeBa/Conv2D/ReadVariableOp�decodeBb/BiasAdd/ReadVariableOp�decodeBb/Conv2D/ReadVariableOp�decodeCa/BiasAdd/ReadVariableOp�decodeCa/Conv2D/ReadVariableOp�decodeCb/BiasAdd/ReadVariableOp�decodeCb/Conv2D/ReadVariableOp�encodeAa/BiasAdd/ReadVariableOp�encodeAa/Conv2D/ReadVariableOp�encodeAb/BiasAdd/ReadVariableOp�encodeAb/Conv2D/ReadVariableOp�encodeBa/BiasAdd/ReadVariableOp�encodeBa/Conv2D/ReadVariableOp�encodeBb/BiasAdd/ReadVariableOp�encodeBb/Conv2D/ReadVariableOp�encodeCa/BiasAdd/ReadVariableOp�encodeCa/Conv2D/ReadVariableOp�encodeCb/BiasAdd/ReadVariableOp�encodeCb/Conv2D/ReadVariableOp�encodeDa/BiasAdd/ReadVariableOp�encodeDa/Conv2D/ReadVariableOp�encodeDb/BiasAdd/ReadVariableOp�encodeDb/Conv2D/ReadVariableOp�encodeEa/BiasAdd/ReadVariableOp�encodeEa/Conv2D/ReadVariableOp�encodeEb/BiasAdd/ReadVariableOp�encodeEb/Conv2D/ReadVariableOp�!transconvA/BiasAdd/ReadVariableOp�*transconvA/conv2d_transpose/ReadVariableOp�!transconvB/BiasAdd/ReadVariableOp�*transconvB/conv2d_transpose/ReadVariableOp�!transconvC/BiasAdd/ReadVariableOp�*transconvC/conv2d_transpose/ReadVariableOp�!transconvE/BiasAdd/ReadVariableOp�*transconvE/conv2d_transpose/ReadVariableOp�
encodeAa/Conv2D/ReadVariableOpReadVariableOp'encodeaa_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
encodeAa/Conv2D/ReadVariableOp�
encodeAa/Conv2DConv2Dinputs&encodeAa/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
2
encodeAa/Conv2D�
encodeAa/BiasAdd/ReadVariableOpReadVariableOp(encodeaa_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
encodeAa/BiasAdd/ReadVariableOp�
encodeAa/BiasAddBiasAddencodeAa/Conv2D:output:0'encodeAa/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������2
encodeAa/BiasAdd}
encodeAa/ReluReluencodeAa/BiasAdd:output:0*
T0*1
_output_shapes
:�����������2
encodeAa/Relu�
encodeAb/Conv2D/ReadVariableOpReadVariableOp'encodeab_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
encodeAb/Conv2D/ReadVariableOp�
encodeAb/Conv2DConv2DencodeAa/Relu:activations:0&encodeAb/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
2
encodeAb/Conv2D�
encodeAb/BiasAdd/ReadVariableOpReadVariableOp(encodeab_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
encodeAb/BiasAdd/ReadVariableOp�
encodeAb/BiasAddBiasAddencodeAb/Conv2D:output:0'encodeAb/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������2
encodeAb/BiasAdd}
encodeAb/ReluReluencodeAb/BiasAdd:output:0*
T0*1
_output_shapes
:�����������2
encodeAb/Relu�
poolA/MaxPoolMaxPoolencodeAb/Relu:activations:0*/
_output_shapes
:���������@@*
ksize
*
paddingVALID*
strides
2
poolA/MaxPool�
encodeBa/Conv2D/ReadVariableOpReadVariableOp'encodeba_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
encodeBa/Conv2D/ReadVariableOp�
encodeBa/Conv2DConv2DpoolA/MaxPool:output:0&encodeBa/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@ *
paddingSAME*
strides
2
encodeBa/Conv2D�
encodeBa/BiasAdd/ReadVariableOpReadVariableOp(encodeba_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
encodeBa/BiasAdd/ReadVariableOp�
encodeBa/BiasAddBiasAddencodeBa/Conv2D:output:0'encodeBa/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@ 2
encodeBa/BiasAdd{
encodeBa/ReluReluencodeBa/BiasAdd:output:0*
T0*/
_output_shapes
:���������@@ 2
encodeBa/Relu�
encodeBb/Conv2D/ReadVariableOpReadVariableOp'encodebb_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02 
encodeBb/Conv2D/ReadVariableOp�
encodeBb/Conv2DConv2DencodeBa/Relu:activations:0&encodeBb/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@ *
paddingSAME*
strides
2
encodeBb/Conv2D�
encodeBb/BiasAdd/ReadVariableOpReadVariableOp(encodebb_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
encodeBb/BiasAdd/ReadVariableOp�
encodeBb/BiasAddBiasAddencodeBb/Conv2D:output:0'encodeBb/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@ 2
encodeBb/BiasAdd{
encodeBb/ReluReluencodeBb/BiasAdd:output:0*
T0*/
_output_shapes
:���������@@ 2
encodeBb/Relu�
poolB/MaxPoolMaxPoolencodeBb/Relu:activations:0*/
_output_shapes
:���������   *
ksize
*
paddingVALID*
strides
2
poolB/MaxPool�
encodeCa/Conv2D/ReadVariableOpReadVariableOp'encodeca_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02 
encodeCa/Conv2D/ReadVariableOp�
encodeCa/Conv2DConv2DpoolB/MaxPool:output:0&encodeCa/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  @*
paddingSAME*
strides
2
encodeCa/Conv2D�
encodeCa/BiasAdd/ReadVariableOpReadVariableOp(encodeca_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
encodeCa/BiasAdd/ReadVariableOp�
encodeCa/BiasAddBiasAddencodeCa/Conv2D:output:0'encodeCa/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  @2
encodeCa/BiasAdd{
encodeCa/ReluReluencodeCa/BiasAdd:output:0*
T0*/
_output_shapes
:���������  @2
encodeCa/Relu�
spatial_dropout2d/IdentityIdentityencodeCa/Relu:activations:0*
T0*/
_output_shapes
:���������  @2
spatial_dropout2d/Identity�
encodeCb/Conv2D/ReadVariableOpReadVariableOp'encodecb_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
encodeCb/Conv2D/ReadVariableOp�
encodeCb/Conv2DConv2D#spatial_dropout2d/Identity:output:0&encodeCb/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  @*
paddingSAME*
strides
2
encodeCb/Conv2D�
encodeCb/BiasAdd/ReadVariableOpReadVariableOp(encodecb_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
encodeCb/BiasAdd/ReadVariableOp�
encodeCb/BiasAddBiasAddencodeCb/Conv2D:output:0'encodeCb/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  @2
encodeCb/BiasAdd{
encodeCb/ReluReluencodeCb/BiasAdd:output:0*
T0*/
_output_shapes
:���������  @2
encodeCb/Relu�
poolC/MaxPoolMaxPoolencodeCb/Relu:activations:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
2
poolC/MaxPool�
encodeDa/Conv2D/ReadVariableOpReadVariableOp'encodeda_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype02 
encodeDa/Conv2D/ReadVariableOp�
encodeDa/Conv2DConv2DpoolC/MaxPool:output:0&encodeDa/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
encodeDa/Conv2D�
encodeDa/BiasAdd/ReadVariableOpReadVariableOp(encodeda_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
encodeDa/BiasAdd/ReadVariableOp�
encodeDa/BiasAddBiasAddencodeDa/Conv2D:output:0'encodeDa/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
encodeDa/BiasAdd|
encodeDa/ReluReluencodeDa/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
encodeDa/Relu�
spatial_dropout2d_1/IdentityIdentityencodeDa/Relu:activations:0*
T0*0
_output_shapes
:����������2
spatial_dropout2d_1/Identity�
encodeDb/Conv2D/ReadVariableOpReadVariableOp'encodedb_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02 
encodeDb/Conv2D/ReadVariableOp�
encodeDb/Conv2DConv2D%spatial_dropout2d_1/Identity:output:0&encodeDb/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
encodeDb/Conv2D�
encodeDb/BiasAdd/ReadVariableOpReadVariableOp(encodedb_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
encodeDb/BiasAdd/ReadVariableOp�
encodeDb/BiasAddBiasAddencodeDb/Conv2D:output:0'encodeDb/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
encodeDb/BiasAdd|
encodeDb/ReluReluencodeDb/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
encodeDb/Relu�
poolD/MaxPoolMaxPoolencodeDb/Relu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2
poolD/MaxPool�
encodeEa/Conv2D/ReadVariableOpReadVariableOp'encodeea_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02 
encodeEa/Conv2D/ReadVariableOp�
encodeEa/Conv2DConv2DpoolD/MaxPool:output:0&encodeEa/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
encodeEa/Conv2D�
encodeEa/BiasAdd/ReadVariableOpReadVariableOp(encodeea_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
encodeEa/BiasAdd/ReadVariableOp�
encodeEa/BiasAddBiasAddencodeEa/Conv2D:output:0'encodeEa/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
encodeEa/BiasAdd|
encodeEa/ReluReluencodeEa/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
encodeEa/Relu�
encodeEb/Conv2D/ReadVariableOpReadVariableOp'encodeeb_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02 
encodeEb/Conv2D/ReadVariableOp�
encodeEb/Conv2DConv2DencodeEa/Relu:activations:0&encodeEb/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
encodeEb/Conv2D�
encodeEb/BiasAdd/ReadVariableOpReadVariableOp(encodeeb_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
encodeEb/BiasAdd/ReadVariableOp�
encodeEb/BiasAddBiasAddencodeEb/Conv2D:output:0'encodeEb/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
encodeEb/BiasAdd|
encodeEb/ReluReluencodeEb/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
encodeEb/Reluo
transconvE/ShapeShapeencodeEb/Relu:activations:0*
T0*
_output_shapes
:2
transconvE/Shape�
transconvE/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
transconvE/strided_slice/stack�
 transconvE/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 transconvE/strided_slice/stack_1�
 transconvE/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 transconvE/strided_slice/stack_2�
transconvE/strided_sliceStridedSlicetransconvE/Shape:output:0'transconvE/strided_slice/stack:output:0)transconvE/strided_slice/stack_1:output:0)transconvE/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
transconvE/strided_slicej
transconvE/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
transconvE/stack/1j
transconvE/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
transconvE/stack/2k
transconvE/stack/3Const*
_output_shapes
: *
dtype0*
value
B :�2
transconvE/stack/3�
transconvE/stackPack!transconvE/strided_slice:output:0transconvE/stack/1:output:0transconvE/stack/2:output:0transconvE/stack/3:output:0*
N*
T0*
_output_shapes
:2
transconvE/stack�
 transconvE/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 transconvE/strided_slice_1/stack�
"transconvE/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"transconvE/strided_slice_1/stack_1�
"transconvE/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"transconvE/strided_slice_1/stack_2�
transconvE/strided_slice_1StridedSlicetransconvE/stack:output:0)transconvE/strided_slice_1/stack:output:0+transconvE/strided_slice_1/stack_1:output:0+transconvE/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
transconvE/strided_slice_1�
*transconvE/conv2d_transpose/ReadVariableOpReadVariableOp3transconve_conv2d_transpose_readvariableop_resource*(
_output_shapes
:��*
dtype02,
*transconvE/conv2d_transpose/ReadVariableOp�
transconvE/conv2d_transposeConv2DBackpropInputtransconvE/stack:output:02transconvE/conv2d_transpose/ReadVariableOp:value:0encodeEb/Relu:activations:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
transconvE/conv2d_transpose�
!transconvE/BiasAdd/ReadVariableOpReadVariableOp*transconve_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02#
!transconvE/BiasAdd/ReadVariableOp�
transconvE/BiasAddBiasAdd$transconvE/conv2d_transpose:output:0)transconvE/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
transconvE/BiasAddl
concatD/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatD/concat/axis�
concatD/concatConcatV2transconvE/BiasAdd:output:0encodeDb/Relu:activations:0concatD/concat/axis:output:0*
N*
T0*0
_output_shapes
:����������2
concatD/concat�
decodeCa/Conv2D/ReadVariableOpReadVariableOp'decodeca_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02 
decodeCa/Conv2D/ReadVariableOp�
decodeCa/Conv2DConv2DconcatD/concat:output:0&decodeCa/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
decodeCa/Conv2D�
decodeCa/BiasAdd/ReadVariableOpReadVariableOp(decodeca_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
decodeCa/BiasAdd/ReadVariableOp�
decodeCa/BiasAddBiasAdddecodeCa/Conv2D:output:0'decodeCa/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
decodeCa/BiasAdd|
decodeCa/ReluReludecodeCa/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
decodeCa/Relu�
decodeCb/Conv2D/ReadVariableOpReadVariableOp'decodecb_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02 
decodeCb/Conv2D/ReadVariableOp�
decodeCb/Conv2DConv2DdecodeCa/Relu:activations:0&decodeCb/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
decodeCb/Conv2D�
decodeCb/BiasAdd/ReadVariableOpReadVariableOp(decodecb_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
decodeCb/BiasAdd/ReadVariableOp�
decodeCb/BiasAddBiasAdddecodeCb/Conv2D:output:0'decodeCb/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
decodeCb/BiasAdd|
decodeCb/ReluReludecodeCb/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
decodeCb/Reluo
transconvC/ShapeShapedecodeCb/Relu:activations:0*
T0*
_output_shapes
:2
transconvC/Shape�
transconvC/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
transconvC/strided_slice/stack�
 transconvC/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 transconvC/strided_slice/stack_1�
 transconvC/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 transconvC/strided_slice/stack_2�
transconvC/strided_sliceStridedSlicetransconvC/Shape:output:0'transconvC/strided_slice/stack:output:0)transconvC/strided_slice/stack_1:output:0)transconvC/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
transconvC/strided_slicej
transconvC/stack/1Const*
_output_shapes
: *
dtype0*
value	B : 2
transconvC/stack/1j
transconvC/stack/2Const*
_output_shapes
: *
dtype0*
value	B : 2
transconvC/stack/2j
transconvC/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2
transconvC/stack/3�
transconvC/stackPack!transconvC/strided_slice:output:0transconvC/stack/1:output:0transconvC/stack/2:output:0transconvC/stack/3:output:0*
N*
T0*
_output_shapes
:2
transconvC/stack�
 transconvC/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 transconvC/strided_slice_1/stack�
"transconvC/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"transconvC/strided_slice_1/stack_1�
"transconvC/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"transconvC/strided_slice_1/stack_2�
transconvC/strided_slice_1StridedSlicetransconvC/stack:output:0)transconvC/strided_slice_1/stack:output:0+transconvC/strided_slice_1/stack_1:output:0+transconvC/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
transconvC/strided_slice_1�
*transconvC/conv2d_transpose/ReadVariableOpReadVariableOp3transconvc_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@�*
dtype02,
*transconvC/conv2d_transpose/ReadVariableOp�
transconvC/conv2d_transposeConv2DBackpropInputtransconvC/stack:output:02transconvC/conv2d_transpose/ReadVariableOp:value:0decodeCb/Relu:activations:0*
T0*/
_output_shapes
:���������  @*
paddingSAME*
strides
2
transconvC/conv2d_transpose�
!transconvC/BiasAdd/ReadVariableOpReadVariableOp*transconvc_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02#
!transconvC/BiasAdd/ReadVariableOp�
transconvC/BiasAddBiasAdd$transconvC/conv2d_transpose:output:0)transconvC/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  @2
transconvC/BiasAddl
concatC/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatC/concat/axis�
concatC/concatConcatV2transconvC/BiasAdd:output:0encodeCb/Relu:activations:0concatC/concat/axis:output:0*
N*
T0*0
_output_shapes
:���������  �2
concatC/concat�
decodeBa/Conv2D/ReadVariableOpReadVariableOp'decodeba_conv2d_readvariableop_resource*'
_output_shapes
:�@*
dtype02 
decodeBa/Conv2D/ReadVariableOp�
decodeBa/Conv2DConv2DconcatC/concat:output:0&decodeBa/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  @*
paddingSAME*
strides
2
decodeBa/Conv2D�
decodeBa/BiasAdd/ReadVariableOpReadVariableOp(decodeba_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
decodeBa/BiasAdd/ReadVariableOp�
decodeBa/BiasAddBiasAdddecodeBa/Conv2D:output:0'decodeBa/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  @2
decodeBa/BiasAdd{
decodeBa/ReluReludecodeBa/BiasAdd:output:0*
T0*/
_output_shapes
:���������  @2
decodeBa/Relu�
decodeBb/Conv2D/ReadVariableOpReadVariableOp'decodebb_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
decodeBb/Conv2D/ReadVariableOp�
decodeBb/Conv2DConv2DdecodeBa/Relu:activations:0&decodeBb/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  @*
paddingSAME*
strides
2
decodeBb/Conv2D�
decodeBb/BiasAdd/ReadVariableOpReadVariableOp(decodebb_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
decodeBb/BiasAdd/ReadVariableOp�
decodeBb/BiasAddBiasAdddecodeBb/Conv2D:output:0'decodeBb/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  @2
decodeBb/BiasAdd{
decodeBb/ReluReludecodeBb/BiasAdd:output:0*
T0*/
_output_shapes
:���������  @2
decodeBb/Reluo
transconvB/ShapeShapedecodeBb/Relu:activations:0*
T0*
_output_shapes
:2
transconvB/Shape�
transconvB/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
transconvB/strided_slice/stack�
 transconvB/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 transconvB/strided_slice/stack_1�
 transconvB/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 transconvB/strided_slice/stack_2�
transconvB/strided_sliceStridedSlicetransconvB/Shape:output:0'transconvB/strided_slice/stack:output:0)transconvB/strided_slice/stack_1:output:0)transconvB/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
transconvB/strided_slicej
transconvB/stack/1Const*
_output_shapes
: *
dtype0*
value	B :@2
transconvB/stack/1j
transconvB/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@2
transconvB/stack/2j
transconvB/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2
transconvB/stack/3�
transconvB/stackPack!transconvB/strided_slice:output:0transconvB/stack/1:output:0transconvB/stack/2:output:0transconvB/stack/3:output:0*
N*
T0*
_output_shapes
:2
transconvB/stack�
 transconvB/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 transconvB/strided_slice_1/stack�
"transconvB/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"transconvB/strided_slice_1/stack_1�
"transconvB/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"transconvB/strided_slice_1/stack_2�
transconvB/strided_slice_1StridedSlicetransconvB/stack:output:0)transconvB/strided_slice_1/stack:output:0+transconvB/strided_slice_1/stack_1:output:0+transconvB/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
transconvB/strided_slice_1�
*transconvB/conv2d_transpose/ReadVariableOpReadVariableOp3transconvb_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype02,
*transconvB/conv2d_transpose/ReadVariableOp�
transconvB/conv2d_transposeConv2DBackpropInputtransconvB/stack:output:02transconvB/conv2d_transpose/ReadVariableOp:value:0decodeBb/Relu:activations:0*
T0*/
_output_shapes
:���������@@ *
paddingSAME*
strides
2
transconvB/conv2d_transpose�
!transconvB/BiasAdd/ReadVariableOpReadVariableOp*transconvb_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02#
!transconvB/BiasAdd/ReadVariableOp�
transconvB/BiasAddBiasAdd$transconvB/conv2d_transpose:output:0)transconvB/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@ 2
transconvB/BiasAddl
concatB/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatB/concat/axis�
concatB/concatConcatV2transconvB/BiasAdd:output:0encodeBb/Relu:activations:0concatB/concat/axis:output:0*
N*
T0*/
_output_shapes
:���������@@@2
concatB/concat�
decodeAa/Conv2D/ReadVariableOpReadVariableOp'decodeaa_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02 
decodeAa/Conv2D/ReadVariableOp�
decodeAa/Conv2DConv2DconcatB/concat:output:0&decodeAa/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@ *
paddingSAME*
strides
2
decodeAa/Conv2D�
decodeAa/BiasAdd/ReadVariableOpReadVariableOp(decodeaa_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
decodeAa/BiasAdd/ReadVariableOp�
decodeAa/BiasAddBiasAdddecodeAa/Conv2D:output:0'decodeAa/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@ 2
decodeAa/BiasAdd{
decodeAa/ReluReludecodeAa/BiasAdd:output:0*
T0*/
_output_shapes
:���������@@ 2
decodeAa/Relu�
decodeAb/Conv2D/ReadVariableOpReadVariableOp'decodeab_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02 
decodeAb/Conv2D/ReadVariableOp�
decodeAb/Conv2DConv2DdecodeAa/Relu:activations:0&decodeAb/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@ *
paddingSAME*
strides
2
decodeAb/Conv2D�
decodeAb/BiasAdd/ReadVariableOpReadVariableOp(decodeab_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
decodeAb/BiasAdd/ReadVariableOp�
decodeAb/BiasAddBiasAdddecodeAb/Conv2D:output:0'decodeAb/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@ 2
decodeAb/BiasAdd{
decodeAb/ReluReludecodeAb/BiasAdd:output:0*
T0*/
_output_shapes
:���������@@ 2
decodeAb/Reluo
transconvA/ShapeShapedecodeAb/Relu:activations:0*
T0*
_output_shapes
:2
transconvA/Shape�
transconvA/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
transconvA/strided_slice/stack�
 transconvA/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 transconvA/strided_slice/stack_1�
 transconvA/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 transconvA/strided_slice/stack_2�
transconvA/strided_sliceStridedSlicetransconvA/Shape:output:0'transconvA/strided_slice/stack:output:0)transconvA/strided_slice/stack_1:output:0)transconvA/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
transconvA/strided_slicek
transconvA/stack/1Const*
_output_shapes
: *
dtype0*
value
B :�2
transconvA/stack/1k
transconvA/stack/2Const*
_output_shapes
: *
dtype0*
value
B :�2
transconvA/stack/2j
transconvA/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
transconvA/stack/3�
transconvA/stackPack!transconvA/strided_slice:output:0transconvA/stack/1:output:0transconvA/stack/2:output:0transconvA/stack/3:output:0*
N*
T0*
_output_shapes
:2
transconvA/stack�
 transconvA/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 transconvA/strided_slice_1/stack�
"transconvA/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"transconvA/strided_slice_1/stack_1�
"transconvA/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"transconvA/strided_slice_1/stack_2�
transconvA/strided_slice_1StridedSlicetransconvA/stack:output:0)transconvA/strided_slice_1/stack:output:0+transconvA/strided_slice_1/stack_1:output:0+transconvA/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
transconvA/strided_slice_1�
*transconvA/conv2d_transpose/ReadVariableOpReadVariableOp3transconva_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02,
*transconvA/conv2d_transpose/ReadVariableOp�
transconvA/conv2d_transposeConv2DBackpropInputtransconvA/stack:output:02transconvA/conv2d_transpose/ReadVariableOp:value:0decodeAb/Relu:activations:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
2
transconvA/conv2d_transpose�
!transconvA/BiasAdd/ReadVariableOpReadVariableOp*transconva_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!transconvA/BiasAdd/ReadVariableOp�
transconvA/BiasAddBiasAdd$transconvA/conv2d_transpose:output:0)transconvA/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������2
transconvA/BiasAddl
concatA/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatA/concat/axis�
concatA/concatConcatV2transconvA/BiasAdd:output:0encodeAb/Relu:activations:0concatA/concat/axis:output:0*
N*
T0*1
_output_shapes
:����������� 2
concatA/concat�
convOuta/Conv2D/ReadVariableOpReadVariableOp'convouta_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
convOuta/Conv2D/ReadVariableOp�
convOuta/Conv2DConv2DconcatA/concat:output:0&convOuta/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
2
convOuta/Conv2D�
convOuta/BiasAdd/ReadVariableOpReadVariableOp(convouta_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
convOuta/BiasAdd/ReadVariableOp�
convOuta/BiasAddBiasAddconvOuta/Conv2D:output:0'convOuta/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������2
convOuta/BiasAdd}
convOuta/ReluReluconvOuta/BiasAdd:output:0*
T0*1
_output_shapes
:�����������2
convOuta/Relu�
convOutb/Conv2D/ReadVariableOpReadVariableOp'convoutb_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
convOutb/Conv2D/ReadVariableOp�
convOutb/Conv2DConv2DconvOuta/Relu:activations:0&convOutb/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
2
convOutb/Conv2D�
convOutb/BiasAdd/ReadVariableOpReadVariableOp(convoutb_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
convOutb/BiasAdd/ReadVariableOp�
convOutb/BiasAddBiasAddconvOutb/Conv2D:output:0'convOutb/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������2
convOutb/BiasAdd}
convOutb/ReluReluconvOutb/BiasAdd:output:0*
T0*1
_output_shapes
:�����������2
convOutb/Relu�
$PredictionMask/Conv2D/ReadVariableOpReadVariableOp-predictionmask_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02&
$PredictionMask/Conv2D/ReadVariableOp�
PredictionMask/Conv2DConv2DconvOutb/Relu:activations:0,PredictionMask/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������*
paddingVALID*
strides
2
PredictionMask/Conv2D�
%PredictionMask/BiasAdd/ReadVariableOpReadVariableOp.predictionmask_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%PredictionMask/BiasAdd/ReadVariableOp�
PredictionMask/BiasAddBiasAddPredictionMask/Conv2D:output:0-PredictionMask/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������2
PredictionMask/BiasAdd�
PredictionMask/SigmoidSigmoidPredictionMask/BiasAdd:output:0*
T0*1
_output_shapes
:�����������2
PredictionMask/Sigmoid�
IdentityIdentityPredictionMask/Sigmoid:y:0&^PredictionMask/BiasAdd/ReadVariableOp%^PredictionMask/Conv2D/ReadVariableOp ^convOuta/BiasAdd/ReadVariableOp^convOuta/Conv2D/ReadVariableOp ^convOutb/BiasAdd/ReadVariableOp^convOutb/Conv2D/ReadVariableOp ^decodeAa/BiasAdd/ReadVariableOp^decodeAa/Conv2D/ReadVariableOp ^decodeAb/BiasAdd/ReadVariableOp^decodeAb/Conv2D/ReadVariableOp ^decodeBa/BiasAdd/ReadVariableOp^decodeBa/Conv2D/ReadVariableOp ^decodeBb/BiasAdd/ReadVariableOp^decodeBb/Conv2D/ReadVariableOp ^decodeCa/BiasAdd/ReadVariableOp^decodeCa/Conv2D/ReadVariableOp ^decodeCb/BiasAdd/ReadVariableOp^decodeCb/Conv2D/ReadVariableOp ^encodeAa/BiasAdd/ReadVariableOp^encodeAa/Conv2D/ReadVariableOp ^encodeAb/BiasAdd/ReadVariableOp^encodeAb/Conv2D/ReadVariableOp ^encodeBa/BiasAdd/ReadVariableOp^encodeBa/Conv2D/ReadVariableOp ^encodeBb/BiasAdd/ReadVariableOp^encodeBb/Conv2D/ReadVariableOp ^encodeCa/BiasAdd/ReadVariableOp^encodeCa/Conv2D/ReadVariableOp ^encodeCb/BiasAdd/ReadVariableOp^encodeCb/Conv2D/ReadVariableOp ^encodeDa/BiasAdd/ReadVariableOp^encodeDa/Conv2D/ReadVariableOp ^encodeDb/BiasAdd/ReadVariableOp^encodeDb/Conv2D/ReadVariableOp ^encodeEa/BiasAdd/ReadVariableOp^encodeEa/Conv2D/ReadVariableOp ^encodeEb/BiasAdd/ReadVariableOp^encodeEb/Conv2D/ReadVariableOp"^transconvA/BiasAdd/ReadVariableOp+^transconvA/conv2d_transpose/ReadVariableOp"^transconvB/BiasAdd/ReadVariableOp+^transconvB/conv2d_transpose/ReadVariableOp"^transconvC/BiasAdd/ReadVariableOp+^transconvC/conv2d_transpose/ReadVariableOp"^transconvE/BiasAdd/ReadVariableOp+^transconvE/conv2d_transpose/ReadVariableOp*
T0*1
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:�����������::::::::::::::::::::::::::::::::::::::::::::::2N
%PredictionMask/BiasAdd/ReadVariableOp%PredictionMask/BiasAdd/ReadVariableOp2L
$PredictionMask/Conv2D/ReadVariableOp$PredictionMask/Conv2D/ReadVariableOp2B
convOuta/BiasAdd/ReadVariableOpconvOuta/BiasAdd/ReadVariableOp2@
convOuta/Conv2D/ReadVariableOpconvOuta/Conv2D/ReadVariableOp2B
convOutb/BiasAdd/ReadVariableOpconvOutb/BiasAdd/ReadVariableOp2@
convOutb/Conv2D/ReadVariableOpconvOutb/Conv2D/ReadVariableOp2B
decodeAa/BiasAdd/ReadVariableOpdecodeAa/BiasAdd/ReadVariableOp2@
decodeAa/Conv2D/ReadVariableOpdecodeAa/Conv2D/ReadVariableOp2B
decodeAb/BiasAdd/ReadVariableOpdecodeAb/BiasAdd/ReadVariableOp2@
decodeAb/Conv2D/ReadVariableOpdecodeAb/Conv2D/ReadVariableOp2B
decodeBa/BiasAdd/ReadVariableOpdecodeBa/BiasAdd/ReadVariableOp2@
decodeBa/Conv2D/ReadVariableOpdecodeBa/Conv2D/ReadVariableOp2B
decodeBb/BiasAdd/ReadVariableOpdecodeBb/BiasAdd/ReadVariableOp2@
decodeBb/Conv2D/ReadVariableOpdecodeBb/Conv2D/ReadVariableOp2B
decodeCa/BiasAdd/ReadVariableOpdecodeCa/BiasAdd/ReadVariableOp2@
decodeCa/Conv2D/ReadVariableOpdecodeCa/Conv2D/ReadVariableOp2B
decodeCb/BiasAdd/ReadVariableOpdecodeCb/BiasAdd/ReadVariableOp2@
decodeCb/Conv2D/ReadVariableOpdecodeCb/Conv2D/ReadVariableOp2B
encodeAa/BiasAdd/ReadVariableOpencodeAa/BiasAdd/ReadVariableOp2@
encodeAa/Conv2D/ReadVariableOpencodeAa/Conv2D/ReadVariableOp2B
encodeAb/BiasAdd/ReadVariableOpencodeAb/BiasAdd/ReadVariableOp2@
encodeAb/Conv2D/ReadVariableOpencodeAb/Conv2D/ReadVariableOp2B
encodeBa/BiasAdd/ReadVariableOpencodeBa/BiasAdd/ReadVariableOp2@
encodeBa/Conv2D/ReadVariableOpencodeBa/Conv2D/ReadVariableOp2B
encodeBb/BiasAdd/ReadVariableOpencodeBb/BiasAdd/ReadVariableOp2@
encodeBb/Conv2D/ReadVariableOpencodeBb/Conv2D/ReadVariableOp2B
encodeCa/BiasAdd/ReadVariableOpencodeCa/BiasAdd/ReadVariableOp2@
encodeCa/Conv2D/ReadVariableOpencodeCa/Conv2D/ReadVariableOp2B
encodeCb/BiasAdd/ReadVariableOpencodeCb/BiasAdd/ReadVariableOp2@
encodeCb/Conv2D/ReadVariableOpencodeCb/Conv2D/ReadVariableOp2B
encodeDa/BiasAdd/ReadVariableOpencodeDa/BiasAdd/ReadVariableOp2@
encodeDa/Conv2D/ReadVariableOpencodeDa/Conv2D/ReadVariableOp2B
encodeDb/BiasAdd/ReadVariableOpencodeDb/BiasAdd/ReadVariableOp2@
encodeDb/Conv2D/ReadVariableOpencodeDb/Conv2D/ReadVariableOp2B
encodeEa/BiasAdd/ReadVariableOpencodeEa/BiasAdd/ReadVariableOp2@
encodeEa/Conv2D/ReadVariableOpencodeEa/Conv2D/ReadVariableOp2B
encodeEb/BiasAdd/ReadVariableOpencodeEb/BiasAdd/ReadVariableOp2@
encodeEb/Conv2D/ReadVariableOpencodeEb/Conv2D/ReadVariableOp2F
!transconvA/BiasAdd/ReadVariableOp!transconvA/BiasAdd/ReadVariableOp2X
*transconvA/conv2d_transpose/ReadVariableOp*transconvA/conv2d_transpose/ReadVariableOp2F
!transconvB/BiasAdd/ReadVariableOp!transconvB/BiasAdd/ReadVariableOp2X
*transconvB/conv2d_transpose/ReadVariableOp*transconvB/conv2d_transpose/ReadVariableOp2F
!transconvC/BiasAdd/ReadVariableOp!transconvC/BiasAdd/ReadVariableOp2X
*transconvC/conv2d_transpose/ReadVariableOp*transconvC/conv2d_transpose/ReadVariableOp2F
!transconvE/BiasAdd/ReadVariableOp!transconvE/BiasAdd/ReadVariableOp2X
*transconvE/conv2d_transpose/ReadVariableOp*transconvE/conv2d_transpose/ReadVariableOp:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
}
(__inference_encodeEb_layer_call_fn_20740

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_encodeEb_layer_call_and_return_conditional_losses_186712
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
}
(__inference_decodeBa_layer_call_fn_20826

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_decodeBa_layer_call_and_return_conditional_losses_187942
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������  @2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:���������  �::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:���������  �
 
_user_specified_nameinputs
�
l
N__inference_spatial_dropout2d_1_layer_call_and_return_conditional_losses_20632

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:����������2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:����������2

Identity_1"!

identity_1Identity_1:output:0*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
C__inference_encodeDa_layer_call_and_return_conditional_losses_18550

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:����������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
n
B__inference_concatD_layer_call_and_return_conditional_losses_20747
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis�
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*0
_output_shapes
:����������2
concatl
IdentityIdentityconcat:output:0*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*]
_input_shapesL
J:,����������������������������:����������:l h
B
_output_shapes0
.:,����������������������������
"
_user_specified_name
inputs/0:ZV
0
_output_shapes
:����������
"
_user_specified_name
inputs/1
�#
�
E__inference_transconvA_layer_call_and_return_conditional_losses_18321

inputs,
(conv2d_transpose_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/3�
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_transpose/ReadVariableOp�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+���������������������������*
paddingSAME*
strides
2
conv2d_transpose�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+��������������������������� ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
S
'__inference_concatA_layer_call_fn_20912
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:����������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_concatA_layer_call_and_return_conditional_losses_189242
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:����������� 2

Identity"
identityIdentity:output:0*]
_input_shapesL
J:+���������������������������:�����������:k g
A
_output_shapes/
-:+���������������������������
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:�����������
"
_user_specified_name
inputs/1
�
�
#__inference_signature_wrapper_19704
mrimages
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallmrimagesunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44*:
Tin3
12/*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*P
_read_only_resource_inputs2
0.	
 !"#$%&'()*+,-.*-
config_proto

CPU

GPU 2J 8� *)
f$R"
 __inference__wrapped_model_179712
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:�����������::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:[ W
1
_output_shapes
:�����������
"
_user_specified_name
MRImages
�
}
(__inference_encodeCa_layer_call_fn_20488

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_encodeCa_layer_call_and_return_conditional_losses_184562
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������  @2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������   ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������   
 
_user_specified_nameinputs
�
�
Q__inference_2DUNet_Brats_Decathlon_layer_call_and_return_conditional_losses_19276

inputs
encodeaa_19150
encodeaa_19152
encodeab_19155
encodeab_19157
encodeba_19161
encodeba_19163
encodebb_19166
encodebb_19168
encodeca_19172
encodeca_19174
encodecb_19178
encodecb_19180
encodeda_19184
encodeda_19186
encodedb_19190
encodedb_19192
encodeea_19196
encodeea_19198
encodeeb_19201
encodeeb_19203
transconve_19206
transconve_19208
decodeca_19212
decodeca_19214
decodecb_19217
decodecb_19219
transconvc_19222
transconvc_19224
decodeba_19228
decodeba_19230
decodebb_19233
decodebb_19235
transconvb_19238
transconvb_19240
decodeaa_19244
decodeaa_19246
decodeab_19249
decodeab_19251
transconva_19254
transconva_19256
convouta_19260
convouta_19262
convoutb_19265
convoutb_19267
predictionmask_19270
predictionmask_19272
identity��&PredictionMask/StatefulPartitionedCall� convOuta/StatefulPartitionedCall� convOutb/StatefulPartitionedCall� decodeAa/StatefulPartitionedCall� decodeAb/StatefulPartitionedCall� decodeBa/StatefulPartitionedCall� decodeBb/StatefulPartitionedCall� decodeCa/StatefulPartitionedCall� decodeCb/StatefulPartitionedCall� encodeAa/StatefulPartitionedCall� encodeAb/StatefulPartitionedCall� encodeBa/StatefulPartitionedCall� encodeBb/StatefulPartitionedCall� encodeCa/StatefulPartitionedCall� encodeCb/StatefulPartitionedCall� encodeDa/StatefulPartitionedCall� encodeDb/StatefulPartitionedCall� encodeEa/StatefulPartitionedCall� encodeEb/StatefulPartitionedCall�)spatial_dropout2d/StatefulPartitionedCall�+spatial_dropout2d_1/StatefulPartitionedCall�"transconvA/StatefulPartitionedCall�"transconvB/StatefulPartitionedCall�"transconvC/StatefulPartitionedCall�"transconvE/StatefulPartitionedCall�
 encodeAa/StatefulPartitionedCallStatefulPartitionedCallinputsencodeaa_19150encodeaa_19152*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_encodeAa_layer_call_and_return_conditional_losses_183462"
 encodeAa/StatefulPartitionedCall�
 encodeAb/StatefulPartitionedCallStatefulPartitionedCall)encodeAa/StatefulPartitionedCall:output:0encodeab_19155encodeab_19157*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_encodeAb_layer_call_and_return_conditional_losses_183732"
 encodeAb/StatefulPartitionedCall�
poolA/PartitionedCallPartitionedCall)encodeAb/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_poolA_layer_call_and_return_conditional_losses_179772
poolA/PartitionedCall�
 encodeBa/StatefulPartitionedCallStatefulPartitionedCallpoolA/PartitionedCall:output:0encodeba_19161encodeba_19163*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_encodeBa_layer_call_and_return_conditional_losses_184012"
 encodeBa/StatefulPartitionedCall�
 encodeBb/StatefulPartitionedCallStatefulPartitionedCall)encodeBa/StatefulPartitionedCall:output:0encodebb_19166encodebb_19168*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_encodeBb_layer_call_and_return_conditional_losses_184282"
 encodeBb/StatefulPartitionedCall�
poolB/PartitionedCallPartitionedCall)encodeBb/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������   * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_poolB_layer_call_and_return_conditional_losses_179892
poolB/PartitionedCall�
 encodeCa/StatefulPartitionedCallStatefulPartitionedCallpoolB/PartitionedCall:output:0encodeca_19172encodeca_19174*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_encodeCa_layer_call_and_return_conditional_losses_184562"
 encodeCa/StatefulPartitionedCall�
)spatial_dropout2d/StatefulPartitionedCallStatefulPartitionedCall)encodeCa/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_spatial_dropout2d_layer_call_and_return_conditional_losses_184942+
)spatial_dropout2d/StatefulPartitionedCall�
 encodeCb/StatefulPartitionedCallStatefulPartitionedCall2spatial_dropout2d/StatefulPartitionedCall:output:0encodecb_19178encodecb_19180*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_encodeCb_layer_call_and_return_conditional_losses_185222"
 encodeCb/StatefulPartitionedCall�
poolC/PartitionedCallPartitionedCall)encodeCb/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_poolC_layer_call_and_return_conditional_losses_180692
poolC/PartitionedCall�
 encodeDa/StatefulPartitionedCallStatefulPartitionedCallpoolC/PartitionedCall:output:0encodeda_19184encodeda_19186*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_encodeDa_layer_call_and_return_conditional_losses_185502"
 encodeDa/StatefulPartitionedCall�
+spatial_dropout2d_1/StatefulPartitionedCallStatefulPartitionedCall)encodeDa/StatefulPartitionedCall:output:0*^spatial_dropout2d/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_spatial_dropout2d_1_layer_call_and_return_conditional_losses_185882-
+spatial_dropout2d_1/StatefulPartitionedCall�
 encodeDb/StatefulPartitionedCallStatefulPartitionedCall4spatial_dropout2d_1/StatefulPartitionedCall:output:0encodedb_19190encodedb_19192*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_encodeDb_layer_call_and_return_conditional_losses_186162"
 encodeDb/StatefulPartitionedCall�
poolD/PartitionedCallPartitionedCall)encodeDb/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_poolD_layer_call_and_return_conditional_losses_181492
poolD/PartitionedCall�
 encodeEa/StatefulPartitionedCallStatefulPartitionedCallpoolD/PartitionedCall:output:0encodeea_19196encodeea_19198*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_encodeEa_layer_call_and_return_conditional_losses_186442"
 encodeEa/StatefulPartitionedCall�
 encodeEb/StatefulPartitionedCallStatefulPartitionedCall)encodeEa/StatefulPartitionedCall:output:0encodeeb_19201encodeeb_19203*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_encodeEb_layer_call_and_return_conditional_losses_186712"
 encodeEb/StatefulPartitionedCall�
"transconvE/StatefulPartitionedCallStatefulPartitionedCall)encodeEb/StatefulPartitionedCall:output:0transconve_19206transconve_19208*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_transconvE_layer_call_and_return_conditional_losses_181892$
"transconvE/StatefulPartitionedCall�
concatD/PartitionedCallPartitionedCall+transconvE/StatefulPartitionedCall:output:0)encodeDb/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_concatD_layer_call_and_return_conditional_losses_186992
concatD/PartitionedCall�
 decodeCa/StatefulPartitionedCallStatefulPartitionedCall concatD/PartitionedCall:output:0decodeca_19212decodeca_19214*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_decodeCa_layer_call_and_return_conditional_losses_187192"
 decodeCa/StatefulPartitionedCall�
 decodeCb/StatefulPartitionedCallStatefulPartitionedCall)decodeCa/StatefulPartitionedCall:output:0decodecb_19217decodecb_19219*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_decodeCb_layer_call_and_return_conditional_losses_187462"
 decodeCb/StatefulPartitionedCall�
"transconvC/StatefulPartitionedCallStatefulPartitionedCall)decodeCb/StatefulPartitionedCall:output:0transconvc_19222transconvc_19224*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_transconvC_layer_call_and_return_conditional_losses_182332$
"transconvC/StatefulPartitionedCall�
concatC/PartitionedCallPartitionedCall+transconvC/StatefulPartitionedCall:output:0)encodeCb/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������  �* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_concatC_layer_call_and_return_conditional_losses_187742
concatC/PartitionedCall�
 decodeBa/StatefulPartitionedCallStatefulPartitionedCall concatC/PartitionedCall:output:0decodeba_19228decodeba_19230*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_decodeBa_layer_call_and_return_conditional_losses_187942"
 decodeBa/StatefulPartitionedCall�
 decodeBb/StatefulPartitionedCallStatefulPartitionedCall)decodeBa/StatefulPartitionedCall:output:0decodebb_19233decodebb_19235*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_decodeBb_layer_call_and_return_conditional_losses_188212"
 decodeBb/StatefulPartitionedCall�
"transconvB/StatefulPartitionedCallStatefulPartitionedCall)decodeBb/StatefulPartitionedCall:output:0transconvb_19238transconvb_19240*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+��������������������������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_transconvB_layer_call_and_return_conditional_losses_182772$
"transconvB/StatefulPartitionedCall�
concatB/PartitionedCallPartitionedCall+transconvB/StatefulPartitionedCall:output:0)encodeBb/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_concatB_layer_call_and_return_conditional_losses_188492
concatB/PartitionedCall�
 decodeAa/StatefulPartitionedCallStatefulPartitionedCall concatB/PartitionedCall:output:0decodeaa_19244decodeaa_19246*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_decodeAa_layer_call_and_return_conditional_losses_188692"
 decodeAa/StatefulPartitionedCall�
 decodeAb/StatefulPartitionedCallStatefulPartitionedCall)decodeAa/StatefulPartitionedCall:output:0decodeab_19249decodeab_19251*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_decodeAb_layer_call_and_return_conditional_losses_188962"
 decodeAb/StatefulPartitionedCall�
"transconvA/StatefulPartitionedCallStatefulPartitionedCall)decodeAb/StatefulPartitionedCall:output:0transconva_19254transconva_19256*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_transconvA_layer_call_and_return_conditional_losses_183212$
"transconvA/StatefulPartitionedCall�
concatA/PartitionedCallPartitionedCall+transconvA/StatefulPartitionedCall:output:0)encodeAb/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:����������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_concatA_layer_call_and_return_conditional_losses_189242
concatA/PartitionedCall�
 convOuta/StatefulPartitionedCallStatefulPartitionedCall concatA/PartitionedCall:output:0convouta_19260convouta_19262*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_convOuta_layer_call_and_return_conditional_losses_189442"
 convOuta/StatefulPartitionedCall�
 convOutb/StatefulPartitionedCallStatefulPartitionedCall)convOuta/StatefulPartitionedCall:output:0convoutb_19265convoutb_19267*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_convOutb_layer_call_and_return_conditional_losses_189712"
 convOutb/StatefulPartitionedCall�
&PredictionMask/StatefulPartitionedCallStatefulPartitionedCall)convOutb/StatefulPartitionedCall:output:0predictionmask_19270predictionmask_19272*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_PredictionMask_layer_call_and_return_conditional_losses_189982(
&PredictionMask/StatefulPartitionedCall�
IdentityIdentity/PredictionMask/StatefulPartitionedCall:output:0'^PredictionMask/StatefulPartitionedCall!^convOuta/StatefulPartitionedCall!^convOutb/StatefulPartitionedCall!^decodeAa/StatefulPartitionedCall!^decodeAb/StatefulPartitionedCall!^decodeBa/StatefulPartitionedCall!^decodeBb/StatefulPartitionedCall!^decodeCa/StatefulPartitionedCall!^decodeCb/StatefulPartitionedCall!^encodeAa/StatefulPartitionedCall!^encodeAb/StatefulPartitionedCall!^encodeBa/StatefulPartitionedCall!^encodeBb/StatefulPartitionedCall!^encodeCa/StatefulPartitionedCall!^encodeCb/StatefulPartitionedCall!^encodeDa/StatefulPartitionedCall!^encodeDb/StatefulPartitionedCall!^encodeEa/StatefulPartitionedCall!^encodeEb/StatefulPartitionedCall*^spatial_dropout2d/StatefulPartitionedCall,^spatial_dropout2d_1/StatefulPartitionedCall#^transconvA/StatefulPartitionedCall#^transconvB/StatefulPartitionedCall#^transconvC/StatefulPartitionedCall#^transconvE/StatefulPartitionedCall*
T0*1
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:�����������::::::::::::::::::::::::::::::::::::::::::::::2P
&PredictionMask/StatefulPartitionedCall&PredictionMask/StatefulPartitionedCall2D
 convOuta/StatefulPartitionedCall convOuta/StatefulPartitionedCall2D
 convOutb/StatefulPartitionedCall convOutb/StatefulPartitionedCall2D
 decodeAa/StatefulPartitionedCall decodeAa/StatefulPartitionedCall2D
 decodeAb/StatefulPartitionedCall decodeAb/StatefulPartitionedCall2D
 decodeBa/StatefulPartitionedCall decodeBa/StatefulPartitionedCall2D
 decodeBb/StatefulPartitionedCall decodeBb/StatefulPartitionedCall2D
 decodeCa/StatefulPartitionedCall decodeCa/StatefulPartitionedCall2D
 decodeCb/StatefulPartitionedCall decodeCb/StatefulPartitionedCall2D
 encodeAa/StatefulPartitionedCall encodeAa/StatefulPartitionedCall2D
 encodeAb/StatefulPartitionedCall encodeAb/StatefulPartitionedCall2D
 encodeBa/StatefulPartitionedCall encodeBa/StatefulPartitionedCall2D
 encodeBb/StatefulPartitionedCall encodeBb/StatefulPartitionedCall2D
 encodeCa/StatefulPartitionedCall encodeCa/StatefulPartitionedCall2D
 encodeCb/StatefulPartitionedCall encodeCb/StatefulPartitionedCall2D
 encodeDa/StatefulPartitionedCall encodeDa/StatefulPartitionedCall2D
 encodeDb/StatefulPartitionedCall encodeDb/StatefulPartitionedCall2D
 encodeEa/StatefulPartitionedCall encodeEa/StatefulPartitionedCall2D
 encodeEb/StatefulPartitionedCall encodeEb/StatefulPartitionedCall2V
)spatial_dropout2d/StatefulPartitionedCall)spatial_dropout2d/StatefulPartitionedCall2Z
+spatial_dropout2d_1/StatefulPartitionedCall+spatial_dropout2d_1/StatefulPartitionedCall2H
"transconvA/StatefulPartitionedCall"transconvA/StatefulPartitionedCall2H
"transconvB/StatefulPartitionedCall"transconvB/StatefulPartitionedCall2H
"transconvC/StatefulPartitionedCall"transconvC/StatefulPartitionedCall2H
"transconvE/StatefulPartitionedCall"transconvE/StatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
6__inference_2DUNet_Brats_Decathlon_layer_call_fn_19371
mrimages
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallmrimagesunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44*:
Tin3
12/*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*P
_read_only_resource_inputs2
0.	
 !"#$%&'()*+,-.*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_2DUNet_Brats_Decathlon_layer_call_and_return_conditional_losses_192762
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:�����������::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:[ W
1
_output_shapes
:�����������
"
_user_specified_name
MRImages
�
k
L__inference_spatial_dropout2d_layer_call_and_return_conditional_losses_20549

inputs
identity�D
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:���������  @2
dropout/Mul�
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/1�
dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/2�
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0'dropout/random_uniform/shape/2:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
dropout/random_uniform/shape�
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*8
_output_shapes&
$:"������������������*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*8
_output_shapes&
$:"������������������2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*8
_output_shapes&
$:"������������������2
dropout/Cast�
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:���������  @2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:���������  @2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������  @:W S
/
_output_shapes
:���������  @
 
_user_specified_nameinputs
�
j
L__inference_spatial_dropout2d_layer_call_and_return_conditional_losses_18060

inputs

identity_1}
IdentityIdentityinputs*
T0*J
_output_shapes8
6:4������������������������������������2

Identity�

Identity_1IdentityIdentity:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity_1"!

identity_1Identity_1:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�

�
C__inference_decodeBb_layer_call_and_return_conditional_losses_20837

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  @*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  @2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������  @2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:���������  @2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������  @::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������  @
 
_user_specified_nameinputs
�
A
%__inference_poolA_layer_call_fn_17983

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_poolA_layer_call_and_return_conditional_losses_179772
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
S
'__inference_concatC_layer_call_fn_20806
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������  �* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_concatC_layer_call_and_return_conditional_losses_187742
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:���������  �2

Identity"
identityIdentity:output:0*[
_input_shapesJ
H:+���������������������������@:���������  @:k g
A
_output_shapes/
-:+���������������������������@
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:���������  @
"
_user_specified_name
inputs/1
�
�
.__inference_PredictionMask_layer_call_fn_20972

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_PredictionMask_layer_call_and_return_conditional_losses_189982
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:�����������::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
n
B__inference_concatC_layer_call_and_return_conditional_losses_20800
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis�
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*0
_output_shapes
:���������  �2
concatl
IdentityIdentityconcat:output:0*
T0*0
_output_shapes
:���������  �2

Identity"
identityIdentity:output:0*[
_input_shapesJ
H:+���������������������������@:���������  @:k g
A
_output_shapes/
-:+���������������������������@
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:���������  @
"
_user_specified_name
inputs/1
�
n
B__inference_concatB_layer_call_and_return_conditional_losses_20853
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis�
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*/
_output_shapes
:���������@@@2
concatk
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:���������@@@2

Identity"
identityIdentity:output:0*[
_input_shapesJ
H:+��������������������������� :���������@@ :k g
A
_output_shapes/
-:+��������������������������� 
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:���������@@ 
"
_user_specified_name
inputs/1
�

�
C__inference_convOuta_layer_call_and_return_conditional_losses_18944

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:�����������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:����������� ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:����������� 
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
G
MRImages;
serving_default_MRImages:0�����������L
PredictionMask:
StatefulPartitionedCall:0�����������tensorflow/serving/predict:��
��
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
	layer-8

layer_with_weights-5

layer-9
layer-10
layer_with_weights-6
layer-11
layer-12
layer_with_weights-7
layer-13
layer-14
layer_with_weights-8
layer-15
layer_with_weights-9
layer-16
layer_with_weights-10
layer-17
layer-18
layer_with_weights-11
layer-19
layer_with_weights-12
layer-20
layer_with_weights-13
layer-21
layer-22
layer_with_weights-14
layer-23
layer_with_weights-15
layer-24
layer_with_weights-16
layer-25
layer-26
layer_with_weights-17
layer-27
layer_with_weights-18
layer-28
layer_with_weights-19
layer-29
layer-30
 layer_with_weights-20
 layer-31
!layer_with_weights-21
!layer-32
"layer_with_weights-22
"layer-33
#	optimizer
$regularization_losses
%	variables
&trainable_variables
'	keras_api
(
signatures
�__call__
�_default_save_signature
+�&call_and_return_all_conditional_losses"��
_tf_keras_network��{"class_name": "Functional", "name": "2DUNet_Brats_Decathlon", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "2DUNet_Brats_Decathlon", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "MRImages"}, "name": "MRImages", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "encodeAa", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "encodeAa", "inbound_nodes": [[["MRImages", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "encodeAb", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "encodeAb", "inbound_nodes": [[["encodeAa", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "poolA", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "poolA", "inbound_nodes": [[["encodeAb", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "encodeBa", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "encodeBa", "inbound_nodes": [[["poolA", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "encodeBb", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "encodeBb", "inbound_nodes": [[["encodeBa", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "poolB", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "poolB", "inbound_nodes": [[["encodeBb", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "encodeCa", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "encodeCa", "inbound_nodes": [[["poolB", 0, 0, {}]]]}, {"class_name": "SpatialDropout2D", "config": {"name": "spatial_dropout2d", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "spatial_dropout2d", "inbound_nodes": [[["encodeCa", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "encodeCb", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "encodeCb", "inbound_nodes": [[["spatial_dropout2d", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "poolC", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "poolC", "inbound_nodes": [[["encodeCb", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "encodeDa", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "encodeDa", "inbound_nodes": [[["poolC", 0, 0, {}]]]}, {"class_name": "SpatialDropout2D", "config": {"name": "spatial_dropout2d_1", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "spatial_dropout2d_1", "inbound_nodes": [[["encodeDa", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "encodeDb", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "encodeDb", "inbound_nodes": [[["spatial_dropout2d_1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "poolD", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "poolD", "inbound_nodes": [[["encodeDb", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "encodeEa", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "encodeEa", "inbound_nodes": [[["poolD", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "encodeEb", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "encodeEb", "inbound_nodes": [[["encodeEa", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "transconvE", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "transconvE", "inbound_nodes": [[["encodeEb", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatD", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatD", "inbound_nodes": [[["transconvE", 0, 0, {}], ["encodeDb", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "decodeCa", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "decodeCa", "inbound_nodes": [[["concatD", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "decodeCb", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "decodeCb", "inbound_nodes": [[["decodeCa", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "transconvC", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "transconvC", "inbound_nodes": [[["decodeCb", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatC", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatC", "inbound_nodes": [[["transconvC", 0, 0, {}], ["encodeCb", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "decodeBa", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "decodeBa", "inbound_nodes": [[["concatC", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "decodeBb", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "decodeBb", "inbound_nodes": [[["decodeBa", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "transconvB", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "transconvB", "inbound_nodes": [[["decodeBb", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatB", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatB", "inbound_nodes": [[["transconvB", 0, 0, {}], ["encodeBb", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "decodeAa", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "decodeAa", "inbound_nodes": [[["concatB", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "decodeAb", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "decodeAb", "inbound_nodes": [[["decodeAa", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "transconvA", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "transconvA", "inbound_nodes": [[["decodeAb", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatA", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatA", "inbound_nodes": [[["transconvA", 0, 0, {}], ["encodeAb", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "convOuta", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "convOuta", "inbound_nodes": [[["concatA", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "convOutb", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "convOutb", "inbound_nodes": [[["convOuta", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "PredictionMask", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "PredictionMask", "inbound_nodes": [[["convOutb", 0, 0, {}]]]}], "input_layers": [["MRImages", 0, 0]], "output_layers": [["PredictionMask", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 128, 128, 1]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 128, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "2DUNet_Brats_Decathlon", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "MRImages"}, "name": "MRImages", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "encodeAa", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "encodeAa", "inbound_nodes": [[["MRImages", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "encodeAb", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "encodeAb", "inbound_nodes": [[["encodeAa", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "poolA", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "poolA", "inbound_nodes": [[["encodeAb", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "encodeBa", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "encodeBa", "inbound_nodes": [[["poolA", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "encodeBb", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "encodeBb", "inbound_nodes": [[["encodeBa", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "poolB", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "poolB", "inbound_nodes": [[["encodeBb", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "encodeCa", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "encodeCa", "inbound_nodes": [[["poolB", 0, 0, {}]]]}, {"class_name": "SpatialDropout2D", "config": {"name": "spatial_dropout2d", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "spatial_dropout2d", "inbound_nodes": [[["encodeCa", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "encodeCb", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "encodeCb", "inbound_nodes": [[["spatial_dropout2d", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "poolC", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "poolC", "inbound_nodes": [[["encodeCb", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "encodeDa", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "encodeDa", "inbound_nodes": [[["poolC", 0, 0, {}]]]}, {"class_name": "SpatialDropout2D", "config": {"name": "spatial_dropout2d_1", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "spatial_dropout2d_1", "inbound_nodes": [[["encodeDa", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "encodeDb", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "encodeDb", "inbound_nodes": [[["spatial_dropout2d_1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "poolD", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "poolD", "inbound_nodes": [[["encodeDb", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "encodeEa", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "encodeEa", "inbound_nodes": [[["poolD", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "encodeEb", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "encodeEb", "inbound_nodes": [[["encodeEa", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "transconvE", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "transconvE", "inbound_nodes": [[["encodeEb", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatD", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatD", "inbound_nodes": [[["transconvE", 0, 0, {}], ["encodeDb", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "decodeCa", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "decodeCa", "inbound_nodes": [[["concatD", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "decodeCb", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "decodeCb", "inbound_nodes": [[["decodeCa", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "transconvC", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "transconvC", "inbound_nodes": [[["decodeCb", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatC", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatC", "inbound_nodes": [[["transconvC", 0, 0, {}], ["encodeCb", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "decodeBa", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "decodeBa", "inbound_nodes": [[["concatC", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "decodeBb", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "decodeBb", "inbound_nodes": [[["decodeBa", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "transconvB", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "transconvB", "inbound_nodes": [[["decodeBb", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatB", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatB", "inbound_nodes": [[["transconvB", 0, 0, {}], ["encodeBb", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "decodeAa", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "decodeAa", "inbound_nodes": [[["concatB", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "decodeAb", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "decodeAb", "inbound_nodes": [[["decodeAa", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "transconvA", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "transconvA", "inbound_nodes": [[["decodeAb", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatA", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatA", "inbound_nodes": [[["transconvA", 0, 0, {}], ["encodeAb", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "convOuta", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "convOuta", "inbound_nodes": [[["concatA", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "convOutb", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "convOutb", "inbound_nodes": [[["convOuta", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "PredictionMask", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "PredictionMask", "inbound_nodes": [[["convOutb", 0, 0, {}]]]}], "input_layers": [["MRImages", 0, 0]], "output_layers": [["PredictionMask", 0, 0]]}}, "training_config": {"loss": "dice_coef_loss", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "dice_coef", "dtype": "float32", "fn": "dice_coef"}}, {"class_name": "MeanMetricWrapper", "config": {"name": "soft_dice_coef", "dtype": "float32", "fn": "soft_dice_coef"}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 9.999999747378752e-05, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
�"�
_tf_keras_input_layer�{"class_name": "InputLayer", "name": "MRImages", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "MRImages"}}
�	

)kernel
*bias
+regularization_losses
,	variables
-trainable_variables
.	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "encodeAa", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "encodeAa", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 128, 1]}}
�	

/kernel
0bias
1regularization_losses
2	variables
3trainable_variables
4	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "encodeAb", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "encodeAb", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 128, 16]}}
�
5regularization_losses
6	variables
7trainable_variables
8	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "MaxPooling2D", "name": "poolA", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "poolA", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�	

9kernel
:bias
;regularization_losses
<	variables
=trainable_variables
>	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "encodeBa", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "encodeBa", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 16]}}
�	

?kernel
@bias
Aregularization_losses
B	variables
Ctrainable_variables
D	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "encodeBb", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "encodeBb", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 32]}}
�
Eregularization_losses
F	variables
Gtrainable_variables
H	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "MaxPooling2D", "name": "poolB", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "poolB", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�	

Ikernel
Jbias
Kregularization_losses
L	variables
Mtrainable_variables
N	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "encodeCa", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "encodeCa", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 32]}}
�
Oregularization_losses
P	variables
Qtrainable_variables
R	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "SpatialDropout2D", "name": "spatial_dropout2d", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "spatial_dropout2d", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�	

Skernel
Tbias
Uregularization_losses
V	variables
Wtrainable_variables
X	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "encodeCb", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "encodeCb", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 64]}}
�
Yregularization_losses
Z	variables
[trainable_variables
\	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "MaxPooling2D", "name": "poolC", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "poolC", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�	

]kernel
^bias
_regularization_losses
`	variables
atrainable_variables
b	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "encodeDa", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "encodeDa", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 16, 64]}}
�
cregularization_losses
d	variables
etrainable_variables
f	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "SpatialDropout2D", "name": "spatial_dropout2d_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "spatial_dropout2d_1", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�	

gkernel
hbias
iregularization_losses
j	variables
ktrainable_variables
l	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "encodeDb", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "encodeDb", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 16, 128]}}
�
mregularization_losses
n	variables
otrainable_variables
p	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "MaxPooling2D", "name": "poolD", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "poolD", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�	

qkernel
rbias
sregularization_losses
t	variables
utrainable_variables
v	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "encodeEa", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "encodeEa", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8, 8, 128]}}
�	

wkernel
xbias
yregularization_losses
z	variables
{trainable_variables
|	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "encodeEb", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "encodeEb", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8, 8, 256]}}
�


}kernel
~bias
regularization_losses
�	variables
�trainable_variables
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Conv2DTranspose", "name": "transconvE", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "transconvE", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8, 8, 256]}}
�
�regularization_losses
�	variables
�trainable_variables
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Concatenate", "name": "concatD", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "concatD", "trainable": true, "dtype": "float32", "axis": -1}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 16, 16, 128]}, {"class_name": "TensorShape", "items": [null, 16, 16, 128]}]}
�	
�kernel
	�bias
�regularization_losses
�	variables
�trainable_variables
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "decodeCa", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "decodeCa", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 16, 256]}}
�	
�kernel
	�bias
�regularization_losses
�	variables
�trainable_variables
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "decodeCb", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "decodeCb", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 16, 128]}}
�

�kernel
	�bias
�regularization_losses
�	variables
�trainable_variables
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Conv2DTranspose", "name": "transconvC", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "transconvC", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 16, 128]}}
�
�regularization_losses
�	variables
�trainable_variables
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Concatenate", "name": "concatC", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "concatC", "trainable": true, "dtype": "float32", "axis": -1}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 32, 32, 64]}, {"class_name": "TensorShape", "items": [null, 32, 32, 64]}]}
�	
�kernel
	�bias
�regularization_losses
�	variables
�trainable_variables
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "decodeBa", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "decodeBa", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 128]}}
�	
�kernel
	�bias
�regularization_losses
�	variables
�trainable_variables
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "decodeBb", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "decodeBb", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 64]}}
�

�kernel
	�bias
�regularization_losses
�	variables
�trainable_variables
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Conv2DTranspose", "name": "transconvB", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "transconvB", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 64]}}
�
�regularization_losses
�	variables
�trainable_variables
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Concatenate", "name": "concatB", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "concatB", "trainable": true, "dtype": "float32", "axis": -1}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 64, 64, 32]}, {"class_name": "TensorShape", "items": [null, 64, 64, 32]}]}
�	
�kernel
	�bias
�regularization_losses
�	variables
�trainable_variables
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "decodeAa", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "decodeAa", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 64]}}
�	
�kernel
	�bias
�regularization_losses
�	variables
�trainable_variables
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "decodeAb", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "decodeAb", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 32]}}
�

�kernel
	�bias
�regularization_losses
�	variables
�trainable_variables
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Conv2DTranspose", "name": "transconvA", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "transconvA", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 32]}}
�
�regularization_losses
�	variables
�trainable_variables
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Concatenate", "name": "concatA", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "concatA", "trainable": true, "dtype": "float32", "axis": -1}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 128, 128, 16]}, {"class_name": "TensorShape", "items": [null, 128, 128, 16]}]}
�	
�kernel
	�bias
�regularization_losses
�	variables
�trainable_variables
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "convOuta", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "convOuta", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 128, 32]}}
�	
�kernel
	�bias
�regularization_losses
�	variables
�trainable_variables
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "convOutb", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "convOutb", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 128, 16]}}
�

�kernel
	�bias
�regularization_losses
�	variables
�trainable_variables
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "PredictionMask", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "PredictionMask", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 128, 16]}}
�
	�iter
�beta_1
�beta_2

�decay
�learning_rate)m�*m�/m�0m�9m�:m�?m�@m�Im�Jm�Sm�Tm�]m�^m�gm�hm�qm�rm�wm�xm�}m�~m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�)v�*v�/v�0v�9v�:v�?v�@v�Iv�Jv�Sv�Tv�]v�^v�gv�hv�qv�rv�wv�xv�}v�~v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�"
	optimizer
 "
trackable_list_wrapper
�
)0
*1
/2
03
94
:5
?6
@7
I8
J9
S10
T11
]12
^13
g14
h15
q16
r17
w18
x19
}20
~21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39
�40
�41
�42
�43
�44
�45"
trackable_list_wrapper
�
)0
*1
/2
03
94
:5
?6
@7
I8
J9
S10
T11
]12
^13
g14
h15
q16
r17
w18
x19
}20
~21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39
�40
�41
�42
�43
�44
�45"
trackable_list_wrapper
�
$regularization_losses
 �layer_regularization_losses
%	variables
�metrics
&trainable_variables
�non_trainable_variables
�layers
�layer_metrics
�__call__
�_default_save_signature
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-
�serving_default"
signature_map
):'2encodeAa/kernel
:2encodeAa/bias
 "
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
�
+regularization_losses
 �layer_regularization_losses
,	variables
�metrics
-trainable_variables
�non_trainable_variables
�layers
�layer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
):'2encodeAb/kernel
:2encodeAb/bias
 "
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
�
1regularization_losses
 �layer_regularization_losses
2	variables
�metrics
3trainable_variables
�non_trainable_variables
�layers
�layer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
5regularization_losses
 �layer_regularization_losses
6	variables
�metrics
7trainable_variables
�non_trainable_variables
�layers
�layer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
):' 2encodeBa/kernel
: 2encodeBa/bias
 "
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
�
;regularization_losses
 �layer_regularization_losses
<	variables
�metrics
=trainable_variables
�non_trainable_variables
�layers
�layer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
):'  2encodeBb/kernel
: 2encodeBb/bias
 "
trackable_list_wrapper
.
?0
@1"
trackable_list_wrapper
.
?0
@1"
trackable_list_wrapper
�
Aregularization_losses
 �layer_regularization_losses
B	variables
�metrics
Ctrainable_variables
�non_trainable_variables
�layers
�layer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
Eregularization_losses
 �layer_regularization_losses
F	variables
�metrics
Gtrainable_variables
�non_trainable_variables
�layers
�layer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
):' @2encodeCa/kernel
:@2encodeCa/bias
 "
trackable_list_wrapper
.
I0
J1"
trackable_list_wrapper
.
I0
J1"
trackable_list_wrapper
�
Kregularization_losses
 �layer_regularization_losses
L	variables
�metrics
Mtrainable_variables
�non_trainable_variables
�layers
�layer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
Oregularization_losses
 �layer_regularization_losses
P	variables
�metrics
Qtrainable_variables
�non_trainable_variables
�layers
�layer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
):'@@2encodeCb/kernel
:@2encodeCb/bias
 "
trackable_list_wrapper
.
S0
T1"
trackable_list_wrapper
.
S0
T1"
trackable_list_wrapper
�
Uregularization_losses
 �layer_regularization_losses
V	variables
�metrics
Wtrainable_variables
�non_trainable_variables
�layers
�layer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
Yregularization_losses
 �layer_regularization_losses
Z	variables
�metrics
[trainable_variables
�non_trainable_variables
�layers
�layer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
*:(@�2encodeDa/kernel
:�2encodeDa/bias
 "
trackable_list_wrapper
.
]0
^1"
trackable_list_wrapper
.
]0
^1"
trackable_list_wrapper
�
_regularization_losses
 �layer_regularization_losses
`	variables
�metrics
atrainable_variables
�non_trainable_variables
�layers
�layer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
cregularization_losses
 �layer_regularization_losses
d	variables
�metrics
etrainable_variables
�non_trainable_variables
�layers
�layer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
+:)��2encodeDb/kernel
:�2encodeDb/bias
 "
trackable_list_wrapper
.
g0
h1"
trackable_list_wrapper
.
g0
h1"
trackable_list_wrapper
�
iregularization_losses
 �layer_regularization_losses
j	variables
�metrics
ktrainable_variables
�non_trainable_variables
�layers
�layer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
mregularization_losses
 �layer_regularization_losses
n	variables
�metrics
otrainable_variables
�non_trainable_variables
�layers
�layer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
+:)��2encodeEa/kernel
:�2encodeEa/bias
 "
trackable_list_wrapper
.
q0
r1"
trackable_list_wrapper
.
q0
r1"
trackable_list_wrapper
�
sregularization_losses
 �layer_regularization_losses
t	variables
�metrics
utrainable_variables
�non_trainable_variables
�layers
�layer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
+:)��2encodeEb/kernel
:�2encodeEb/bias
 "
trackable_list_wrapper
.
w0
x1"
trackable_list_wrapper
.
w0
x1"
trackable_list_wrapper
�
yregularization_losses
 �layer_regularization_losses
z	variables
�metrics
{trainable_variables
�non_trainable_variables
�layers
�layer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-:+��2transconvE/kernel
:�2transconvE/bias
 "
trackable_list_wrapper
.
}0
~1"
trackable_list_wrapper
.
}0
~1"
trackable_list_wrapper
�
regularization_losses
 �layer_regularization_losses
�	variables
�metrics
�trainable_variables
�non_trainable_variables
�layers
�layer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�regularization_losses
 �layer_regularization_losses
�	variables
�metrics
�trainable_variables
�non_trainable_variables
�layers
�layer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
+:)��2decodeCa/kernel
:�2decodeCa/bias
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
�
�regularization_losses
 �layer_regularization_losses
�	variables
�metrics
�trainable_variables
�non_trainable_variables
�layers
�layer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
+:)��2decodeCb/kernel
:�2decodeCb/bias
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
�
�regularization_losses
 �layer_regularization_losses
�	variables
�metrics
�trainable_variables
�non_trainable_variables
�layers
�layer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
,:*@�2transconvC/kernel
:@2transconvC/bias
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
�
�regularization_losses
 �layer_regularization_losses
�	variables
�metrics
�trainable_variables
�non_trainable_variables
�layers
�layer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�regularization_losses
 �layer_regularization_losses
�	variables
�metrics
�trainable_variables
�non_trainable_variables
�layers
�layer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
*:(�@2decodeBa/kernel
:@2decodeBa/bias
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
�
�regularization_losses
 �layer_regularization_losses
�	variables
�metrics
�trainable_variables
�non_trainable_variables
�layers
�layer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
):'@@2decodeBb/kernel
:@2decodeBb/bias
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
�
�regularization_losses
 �layer_regularization_losses
�	variables
�metrics
�trainable_variables
�non_trainable_variables
�layers
�layer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
+:) @2transconvB/kernel
: 2transconvB/bias
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
�
�regularization_losses
 �layer_regularization_losses
�	variables
�metrics
�trainable_variables
�non_trainable_variables
�layers
�layer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�regularization_losses
 �layer_regularization_losses
�	variables
�metrics
�trainable_variables
�non_trainable_variables
�layers
�layer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
):'@ 2decodeAa/kernel
: 2decodeAa/bias
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
�
�regularization_losses
 �layer_regularization_losses
�	variables
�metrics
�trainable_variables
�non_trainable_variables
�layers
�layer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
):'  2decodeAb/kernel
: 2decodeAb/bias
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
�
�regularization_losses
 �layer_regularization_losses
�	variables
�metrics
�trainable_variables
�non_trainable_variables
�layers
�layer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
+:) 2transconvA/kernel
:2transconvA/bias
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
�
�regularization_losses
 �layer_regularization_losses
�	variables
�metrics
�trainable_variables
�non_trainable_variables
�layers
�layer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�regularization_losses
 �layer_regularization_losses
�	variables
�metrics
�trainable_variables
�non_trainable_variables
�layers
�layer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
):' 2convOuta/kernel
:2convOuta/bias
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
�
�regularization_losses
 �layer_regularization_losses
�	variables
�metrics
�trainable_variables
�non_trainable_variables
�layers
�layer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
):'2convOutb/kernel
:2convOutb/bias
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
�
�regularization_losses
 �layer_regularization_losses
�	variables
�metrics
�trainable_variables
�non_trainable_variables
�layers
�layer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
/:-2PredictionMask/kernel
!:2PredictionMask/bias
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
�
�regularization_losses
 �layer_regularization_losses
�	variables
�metrics
�trainable_variables
�non_trainable_variables
�layers
�layer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
8
�0
�1
�2"
trackable_list_wrapper
 "
trackable_list_wrapper
�
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
 31
!32
"33"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�

�total

�count
�	variables
�	keras_api"�
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
�

�total

�count
�
_fn_kwargs
�	variables
�	keras_api"�
_tf_keras_metric�{"class_name": "MeanMetricWrapper", "name": "dice_coef", "dtype": "float32", "config": {"name": "dice_coef", "dtype": "float32", "fn": "dice_coef"}}
�

�total

�count
�
_fn_kwargs
�	variables
�	keras_api"�
_tf_keras_metric�{"class_name": "MeanMetricWrapper", "name": "soft_dice_coef", "dtype": "float32", "config": {"name": "soft_dice_coef", "dtype": "float32", "fn": "soft_dice_coef"}}
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
.:,2Adam/encodeAa/kernel/m
 :2Adam/encodeAa/bias/m
.:,2Adam/encodeAb/kernel/m
 :2Adam/encodeAb/bias/m
.:, 2Adam/encodeBa/kernel/m
 : 2Adam/encodeBa/bias/m
.:,  2Adam/encodeBb/kernel/m
 : 2Adam/encodeBb/bias/m
.:, @2Adam/encodeCa/kernel/m
 :@2Adam/encodeCa/bias/m
.:,@@2Adam/encodeCb/kernel/m
 :@2Adam/encodeCb/bias/m
/:-@�2Adam/encodeDa/kernel/m
!:�2Adam/encodeDa/bias/m
0:.��2Adam/encodeDb/kernel/m
!:�2Adam/encodeDb/bias/m
0:.��2Adam/encodeEa/kernel/m
!:�2Adam/encodeEa/bias/m
0:.��2Adam/encodeEb/kernel/m
!:�2Adam/encodeEb/bias/m
2:0��2Adam/transconvE/kernel/m
#:!�2Adam/transconvE/bias/m
0:.��2Adam/decodeCa/kernel/m
!:�2Adam/decodeCa/bias/m
0:.��2Adam/decodeCb/kernel/m
!:�2Adam/decodeCb/bias/m
1:/@�2Adam/transconvC/kernel/m
": @2Adam/transconvC/bias/m
/:-�@2Adam/decodeBa/kernel/m
 :@2Adam/decodeBa/bias/m
.:,@@2Adam/decodeBb/kernel/m
 :@2Adam/decodeBb/bias/m
0:. @2Adam/transconvB/kernel/m
":  2Adam/transconvB/bias/m
.:,@ 2Adam/decodeAa/kernel/m
 : 2Adam/decodeAa/bias/m
.:,  2Adam/decodeAb/kernel/m
 : 2Adam/decodeAb/bias/m
0:. 2Adam/transconvA/kernel/m
": 2Adam/transconvA/bias/m
.:, 2Adam/convOuta/kernel/m
 :2Adam/convOuta/bias/m
.:,2Adam/convOutb/kernel/m
 :2Adam/convOutb/bias/m
4:22Adam/PredictionMask/kernel/m
&:$2Adam/PredictionMask/bias/m
.:,2Adam/encodeAa/kernel/v
 :2Adam/encodeAa/bias/v
.:,2Adam/encodeAb/kernel/v
 :2Adam/encodeAb/bias/v
.:, 2Adam/encodeBa/kernel/v
 : 2Adam/encodeBa/bias/v
.:,  2Adam/encodeBb/kernel/v
 : 2Adam/encodeBb/bias/v
.:, @2Adam/encodeCa/kernel/v
 :@2Adam/encodeCa/bias/v
.:,@@2Adam/encodeCb/kernel/v
 :@2Adam/encodeCb/bias/v
/:-@�2Adam/encodeDa/kernel/v
!:�2Adam/encodeDa/bias/v
0:.��2Adam/encodeDb/kernel/v
!:�2Adam/encodeDb/bias/v
0:.��2Adam/encodeEa/kernel/v
!:�2Adam/encodeEa/bias/v
0:.��2Adam/encodeEb/kernel/v
!:�2Adam/encodeEb/bias/v
2:0��2Adam/transconvE/kernel/v
#:!�2Adam/transconvE/bias/v
0:.��2Adam/decodeCa/kernel/v
!:�2Adam/decodeCa/bias/v
0:.��2Adam/decodeCb/kernel/v
!:�2Adam/decodeCb/bias/v
1:/@�2Adam/transconvC/kernel/v
": @2Adam/transconvC/bias/v
/:-�@2Adam/decodeBa/kernel/v
 :@2Adam/decodeBa/bias/v
.:,@@2Adam/decodeBb/kernel/v
 :@2Adam/decodeBb/bias/v
0:. @2Adam/transconvB/kernel/v
":  2Adam/transconvB/bias/v
.:,@ 2Adam/decodeAa/kernel/v
 : 2Adam/decodeAa/bias/v
.:,  2Adam/decodeAb/kernel/v
 : 2Adam/decodeAb/bias/v
0:. 2Adam/transconvA/kernel/v
": 2Adam/transconvA/bias/v
.:, 2Adam/convOuta/kernel/v
 :2Adam/convOuta/bias/v
.:,2Adam/convOutb/kernel/v
 :2Adam/convOutb/bias/v
4:22Adam/PredictionMask/kernel/v
&:$2Adam/PredictionMask/bias/v
�2�
6__inference_2DUNet_Brats_Decathlon_layer_call_fn_20291
6__inference_2DUNet_Brats_Decathlon_layer_call_fn_19597
6__inference_2DUNet_Brats_Decathlon_layer_call_fn_19371
6__inference_2DUNet_Brats_Decathlon_layer_call_fn_20388�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
 __inference__wrapped_model_17971�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *1�.
,�)
MRImages�����������
�2�
Q__inference_2DUNet_Brats_Decathlon_layer_call_and_return_conditional_losses_19967
Q__inference_2DUNet_Brats_Decathlon_layer_call_and_return_conditional_losses_19144
Q__inference_2DUNet_Brats_Decathlon_layer_call_and_return_conditional_losses_20194
Q__inference_2DUNet_Brats_Decathlon_layer_call_and_return_conditional_losses_19015�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
(__inference_encodeAa_layer_call_fn_20408�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_encodeAa_layer_call_and_return_conditional_losses_20399�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
(__inference_encodeAb_layer_call_fn_20428�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_encodeAb_layer_call_and_return_conditional_losses_20419�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
%__inference_poolA_layer_call_fn_17983�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
@__inference_poolA_layer_call_and_return_conditional_losses_17977�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
(__inference_encodeBa_layer_call_fn_20448�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_encodeBa_layer_call_and_return_conditional_losses_20439�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
(__inference_encodeBb_layer_call_fn_20468�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_encodeBb_layer_call_and_return_conditional_losses_20459�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
%__inference_poolB_layer_call_fn_17995�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
@__inference_poolB_layer_call_and_return_conditional_losses_17989�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
(__inference_encodeCa_layer_call_fn_20488�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_encodeCa_layer_call_and_return_conditional_losses_20479�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
1__inference_spatial_dropout2d_layer_call_fn_20564
1__inference_spatial_dropout2d_layer_call_fn_20526
1__inference_spatial_dropout2d_layer_call_fn_20559
1__inference_spatial_dropout2d_layer_call_fn_20521�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
L__inference_spatial_dropout2d_layer_call_and_return_conditional_losses_20516
L__inference_spatial_dropout2d_layer_call_and_return_conditional_losses_20554
L__inference_spatial_dropout2d_layer_call_and_return_conditional_losses_20511
L__inference_spatial_dropout2d_layer_call_and_return_conditional_losses_20549�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
(__inference_encodeCb_layer_call_fn_20584�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_encodeCb_layer_call_and_return_conditional_losses_20575�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
%__inference_poolC_layer_call_fn_18075�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
@__inference_poolC_layer_call_and_return_conditional_losses_18069�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
(__inference_encodeDa_layer_call_fn_20604�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_encodeDa_layer_call_and_return_conditional_losses_20595�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
3__inference_spatial_dropout2d_1_layer_call_fn_20637
3__inference_spatial_dropout2d_1_layer_call_fn_20675
3__inference_spatial_dropout2d_1_layer_call_fn_20642
3__inference_spatial_dropout2d_1_layer_call_fn_20680�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
N__inference_spatial_dropout2d_1_layer_call_and_return_conditional_losses_20665
N__inference_spatial_dropout2d_1_layer_call_and_return_conditional_losses_20627
N__inference_spatial_dropout2d_1_layer_call_and_return_conditional_losses_20632
N__inference_spatial_dropout2d_1_layer_call_and_return_conditional_losses_20670�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
(__inference_encodeDb_layer_call_fn_20700�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_encodeDb_layer_call_and_return_conditional_losses_20691�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
%__inference_poolD_layer_call_fn_18155�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
@__inference_poolD_layer_call_and_return_conditional_losses_18149�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
(__inference_encodeEa_layer_call_fn_20720�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_encodeEa_layer_call_and_return_conditional_losses_20711�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
(__inference_encodeEb_layer_call_fn_20740�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_encodeEb_layer_call_and_return_conditional_losses_20731�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
*__inference_transconvE_layer_call_fn_18199�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *8�5
3�0,����������������������������
�2�
E__inference_transconvE_layer_call_and_return_conditional_losses_18189�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *8�5
3�0,����������������������������
�2�
'__inference_concatD_layer_call_fn_20753�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
B__inference_concatD_layer_call_and_return_conditional_losses_20747�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
(__inference_decodeCa_layer_call_fn_20773�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_decodeCa_layer_call_and_return_conditional_losses_20764�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
(__inference_decodeCb_layer_call_fn_20793�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_decodeCb_layer_call_and_return_conditional_losses_20784�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
*__inference_transconvC_layer_call_fn_18243�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *8�5
3�0,����������������������������
�2�
E__inference_transconvC_layer_call_and_return_conditional_losses_18233�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *8�5
3�0,����������������������������
�2�
'__inference_concatC_layer_call_fn_20806�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
B__inference_concatC_layer_call_and_return_conditional_losses_20800�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
(__inference_decodeBa_layer_call_fn_20826�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_decodeBa_layer_call_and_return_conditional_losses_20817�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
(__inference_decodeBb_layer_call_fn_20846�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_decodeBb_layer_call_and_return_conditional_losses_20837�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
*__inference_transconvB_layer_call_fn_18287�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+���������������������������@
�2�
E__inference_transconvB_layer_call_and_return_conditional_losses_18277�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+���������������������������@
�2�
'__inference_concatB_layer_call_fn_20859�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
B__inference_concatB_layer_call_and_return_conditional_losses_20853�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
(__inference_decodeAa_layer_call_fn_20879�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_decodeAa_layer_call_and_return_conditional_losses_20870�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
(__inference_decodeAb_layer_call_fn_20899�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_decodeAb_layer_call_and_return_conditional_losses_20890�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
*__inference_transconvA_layer_call_fn_18331�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+��������������������������� 
�2�
E__inference_transconvA_layer_call_and_return_conditional_losses_18321�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+��������������������������� 
�2�
'__inference_concatA_layer_call_fn_20912�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
B__inference_concatA_layer_call_and_return_conditional_losses_20906�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
(__inference_convOuta_layer_call_fn_20932�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_convOuta_layer_call_and_return_conditional_losses_20923�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
(__inference_convOutb_layer_call_fn_20952�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_convOutb_layer_call_and_return_conditional_losses_20943�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
.__inference_PredictionMask_layer_call_fn_20972�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
I__inference_PredictionMask_layer_call_and_return_conditional_losses_20963�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
#__inference_signature_wrapper_19704MRImages"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 �
Q__inference_2DUNet_Brats_Decathlon_layer_call_and_return_conditional_losses_19015�F)*/09:?@IJST]^ghqrwx}~������������������������C�@
9�6
,�)
MRImages�����������
p

 
� "/�,
%�"
0�����������
� �
Q__inference_2DUNet_Brats_Decathlon_layer_call_and_return_conditional_losses_19144�F)*/09:?@IJST]^ghqrwx}~������������������������C�@
9�6
,�)
MRImages�����������
p 

 
� "/�,
%�"
0�����������
� �
Q__inference_2DUNet_Brats_Decathlon_layer_call_and_return_conditional_losses_19967�F)*/09:?@IJST]^ghqrwx}~������������������������A�>
7�4
*�'
inputs�����������
p

 
� "/�,
%�"
0�����������
� �
Q__inference_2DUNet_Brats_Decathlon_layer_call_and_return_conditional_losses_20194�F)*/09:?@IJST]^ghqrwx}~������������������������A�>
7�4
*�'
inputs�����������
p 

 
� "/�,
%�"
0�����������
� �
6__inference_2DUNet_Brats_Decathlon_layer_call_fn_19371�F)*/09:?@IJST]^ghqrwx}~������������������������C�@
9�6
,�)
MRImages�����������
p

 
� ""�������������
6__inference_2DUNet_Brats_Decathlon_layer_call_fn_19597�F)*/09:?@IJST]^ghqrwx}~������������������������C�@
9�6
,�)
MRImages�����������
p 

 
� ""�������������
6__inference_2DUNet_Brats_Decathlon_layer_call_fn_20291�F)*/09:?@IJST]^ghqrwx}~������������������������A�>
7�4
*�'
inputs�����������
p

 
� ""�������������
6__inference_2DUNet_Brats_Decathlon_layer_call_fn_20388�F)*/09:?@IJST]^ghqrwx}~������������������������A�>
7�4
*�'
inputs�����������
p 

 
� ""�������������
I__inference_PredictionMask_layer_call_and_return_conditional_losses_20963r��9�6
/�,
*�'
inputs�����������
� "/�,
%�"
0�����������
� �
.__inference_PredictionMask_layer_call_fn_20972e��9�6
/�,
*�'
inputs�����������
� ""�������������
 __inference__wrapped_model_17971�F)*/09:?@IJST]^ghqrwx}~������������������������;�8
1�.
,�)
MRImages�����������
� "I�F
D
PredictionMask2�/
PredictionMask������������
B__inference_concatA_layer_call_and_return_conditional_losses_20906�~�{
t�q
o�l
<�9
inputs/0+���������������������������
,�)
inputs/1�����������
� "/�,
%�"
0����������� 
� �
'__inference_concatA_layer_call_fn_20912�~�{
t�q
o�l
<�9
inputs/0+���������������������������
,�)
inputs/1�����������
� ""������������ �
B__inference_concatB_layer_call_and_return_conditional_losses_20853�|�y
r�o
m�j
<�9
inputs/0+��������������������������� 
*�'
inputs/1���������@@ 
� "-�*
#� 
0���������@@@
� �
'__inference_concatB_layer_call_fn_20859�|�y
r�o
m�j
<�9
inputs/0+��������������������������� 
*�'
inputs/1���������@@ 
� " ����������@@@�
B__inference_concatC_layer_call_and_return_conditional_losses_20800�|�y
r�o
m�j
<�9
inputs/0+���������������������������@
*�'
inputs/1���������  @
� ".�+
$�!
0���������  �
� �
'__inference_concatC_layer_call_fn_20806�|�y
r�o
m�j
<�9
inputs/0+���������������������������@
*�'
inputs/1���������  @
� "!����������  ��
B__inference_concatD_layer_call_and_return_conditional_losses_20747�~�{
t�q
o�l
=�:
inputs/0,����������������������������
+�(
inputs/1����������
� ".�+
$�!
0����������
� �
'__inference_concatD_layer_call_fn_20753�~�{
t�q
o�l
=�:
inputs/0,����������������������������
+�(
inputs/1����������
� "!������������
C__inference_convOuta_layer_call_and_return_conditional_losses_20923r��9�6
/�,
*�'
inputs����������� 
� "/�,
%�"
0�����������
� �
(__inference_convOuta_layer_call_fn_20932e��9�6
/�,
*�'
inputs����������� 
� ""�������������
C__inference_convOutb_layer_call_and_return_conditional_losses_20943r��9�6
/�,
*�'
inputs�����������
� "/�,
%�"
0�����������
� �
(__inference_convOutb_layer_call_fn_20952e��9�6
/�,
*�'
inputs�����������
� ""�������������
C__inference_decodeAa_layer_call_and_return_conditional_losses_20870n��7�4
-�*
(�%
inputs���������@@@
� "-�*
#� 
0���������@@ 
� �
(__inference_decodeAa_layer_call_fn_20879a��7�4
-�*
(�%
inputs���������@@@
� " ����������@@ �
C__inference_decodeAb_layer_call_and_return_conditional_losses_20890n��7�4
-�*
(�%
inputs���������@@ 
� "-�*
#� 
0���������@@ 
� �
(__inference_decodeAb_layer_call_fn_20899a��7�4
-�*
(�%
inputs���������@@ 
� " ����������@@ �
C__inference_decodeBa_layer_call_and_return_conditional_losses_20817o��8�5
.�+
)�&
inputs���������  �
� "-�*
#� 
0���������  @
� �
(__inference_decodeBa_layer_call_fn_20826b��8�5
.�+
)�&
inputs���������  �
� " ����������  @�
C__inference_decodeBb_layer_call_and_return_conditional_losses_20837n��7�4
-�*
(�%
inputs���������  @
� "-�*
#� 
0���������  @
� �
(__inference_decodeBb_layer_call_fn_20846a��7�4
-�*
(�%
inputs���������  @
� " ����������  @�
C__inference_decodeCa_layer_call_and_return_conditional_losses_20764p��8�5
.�+
)�&
inputs����������
� ".�+
$�!
0����������
� �
(__inference_decodeCa_layer_call_fn_20773c��8�5
.�+
)�&
inputs����������
� "!������������
C__inference_decodeCb_layer_call_and_return_conditional_losses_20784p��8�5
.�+
)�&
inputs����������
� ".�+
$�!
0����������
� �
(__inference_decodeCb_layer_call_fn_20793c��8�5
.�+
)�&
inputs����������
� "!������������
C__inference_encodeAa_layer_call_and_return_conditional_losses_20399p)*9�6
/�,
*�'
inputs�����������
� "/�,
%�"
0�����������
� �
(__inference_encodeAa_layer_call_fn_20408c)*9�6
/�,
*�'
inputs�����������
� ""�������������
C__inference_encodeAb_layer_call_and_return_conditional_losses_20419p/09�6
/�,
*�'
inputs�����������
� "/�,
%�"
0�����������
� �
(__inference_encodeAb_layer_call_fn_20428c/09�6
/�,
*�'
inputs�����������
� ""�������������
C__inference_encodeBa_layer_call_and_return_conditional_losses_20439l9:7�4
-�*
(�%
inputs���������@@
� "-�*
#� 
0���������@@ 
� �
(__inference_encodeBa_layer_call_fn_20448_9:7�4
-�*
(�%
inputs���������@@
� " ����������@@ �
C__inference_encodeBb_layer_call_and_return_conditional_losses_20459l?@7�4
-�*
(�%
inputs���������@@ 
� "-�*
#� 
0���������@@ 
� �
(__inference_encodeBb_layer_call_fn_20468_?@7�4
-�*
(�%
inputs���������@@ 
� " ����������@@ �
C__inference_encodeCa_layer_call_and_return_conditional_losses_20479lIJ7�4
-�*
(�%
inputs���������   
� "-�*
#� 
0���������  @
� �
(__inference_encodeCa_layer_call_fn_20488_IJ7�4
-�*
(�%
inputs���������   
� " ����������  @�
C__inference_encodeCb_layer_call_and_return_conditional_losses_20575lST7�4
-�*
(�%
inputs���������  @
� "-�*
#� 
0���������  @
� �
(__inference_encodeCb_layer_call_fn_20584_ST7�4
-�*
(�%
inputs���������  @
� " ����������  @�
C__inference_encodeDa_layer_call_and_return_conditional_losses_20595m]^7�4
-�*
(�%
inputs���������@
� ".�+
$�!
0����������
� �
(__inference_encodeDa_layer_call_fn_20604`]^7�4
-�*
(�%
inputs���������@
� "!������������
C__inference_encodeDb_layer_call_and_return_conditional_losses_20691ngh8�5
.�+
)�&
inputs����������
� ".�+
$�!
0����������
� �
(__inference_encodeDb_layer_call_fn_20700agh8�5
.�+
)�&
inputs����������
� "!������������
C__inference_encodeEa_layer_call_and_return_conditional_losses_20711nqr8�5
.�+
)�&
inputs����������
� ".�+
$�!
0����������
� �
(__inference_encodeEa_layer_call_fn_20720aqr8�5
.�+
)�&
inputs����������
� "!������������
C__inference_encodeEb_layer_call_and_return_conditional_losses_20731nwx8�5
.�+
)�&
inputs����������
� ".�+
$�!
0����������
� �
(__inference_encodeEb_layer_call_fn_20740awx8�5
.�+
)�&
inputs����������
� "!������������
@__inference_poolA_layer_call_and_return_conditional_losses_17977�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
%__inference_poolA_layer_call_fn_17983�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
@__inference_poolB_layer_call_and_return_conditional_losses_17989�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
%__inference_poolB_layer_call_fn_17995�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
@__inference_poolC_layer_call_and_return_conditional_losses_18069�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
%__inference_poolC_layer_call_fn_18075�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
@__inference_poolD_layer_call_and_return_conditional_losses_18149�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
%__inference_poolD_layer_call_fn_18155�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
#__inference_signature_wrapper_19704�F)*/09:?@IJST]^ghqrwx}~������������������������G�D
� 
=�:
8
MRImages,�)
MRImages�����������"I�F
D
PredictionMask2�/
PredictionMask������������
N__inference_spatial_dropout2d_1_layer_call_and_return_conditional_losses_20627n<�9
2�/
)�&
inputs����������
p
� ".�+
$�!
0����������
� �
N__inference_spatial_dropout2d_1_layer_call_and_return_conditional_losses_20632n<�9
2�/
)�&
inputs����������
p 
� ".�+
$�!
0����������
� �
N__inference_spatial_dropout2d_1_layer_call_and_return_conditional_losses_20665�V�S
L�I
C�@
inputs4������������������������������������
p
� "H�E
>�;
04������������������������������������
� �
N__inference_spatial_dropout2d_1_layer_call_and_return_conditional_losses_20670�V�S
L�I
C�@
inputs4������������������������������������
p 
� "H�E
>�;
04������������������������������������
� �
3__inference_spatial_dropout2d_1_layer_call_fn_20637a<�9
2�/
)�&
inputs����������
p
� "!������������
3__inference_spatial_dropout2d_1_layer_call_fn_20642a<�9
2�/
)�&
inputs����������
p 
� "!������������
3__inference_spatial_dropout2d_1_layer_call_fn_20675�V�S
L�I
C�@
inputs4������������������������������������
p
� ";�84�������������������������������������
3__inference_spatial_dropout2d_1_layer_call_fn_20680�V�S
L�I
C�@
inputs4������������������������������������
p 
� ";�84�������������������������������������
L__inference_spatial_dropout2d_layer_call_and_return_conditional_losses_20511�V�S
L�I
C�@
inputs4������������������������������������
p
� "H�E
>�;
04������������������������������������
� �
L__inference_spatial_dropout2d_layer_call_and_return_conditional_losses_20516�V�S
L�I
C�@
inputs4������������������������������������
p 
� "H�E
>�;
04������������������������������������
� �
L__inference_spatial_dropout2d_layer_call_and_return_conditional_losses_20549l;�8
1�.
(�%
inputs���������  @
p
� "-�*
#� 
0���������  @
� �
L__inference_spatial_dropout2d_layer_call_and_return_conditional_losses_20554l;�8
1�.
(�%
inputs���������  @
p 
� "-�*
#� 
0���������  @
� �
1__inference_spatial_dropout2d_layer_call_fn_20521�V�S
L�I
C�@
inputs4������������������������������������
p
� ";�84�������������������������������������
1__inference_spatial_dropout2d_layer_call_fn_20526�V�S
L�I
C�@
inputs4������������������������������������
p 
� ";�84�������������������������������������
1__inference_spatial_dropout2d_layer_call_fn_20559_;�8
1�.
(�%
inputs���������  @
p
� " ����������  @�
1__inference_spatial_dropout2d_layer_call_fn_20564_;�8
1�.
(�%
inputs���������  @
p 
� " ����������  @�
E__inference_transconvA_layer_call_and_return_conditional_losses_18321���I�F
?�<
:�7
inputs+��������������������������� 
� "?�<
5�2
0+���������������������������
� �
*__inference_transconvA_layer_call_fn_18331���I�F
?�<
:�7
inputs+��������������������������� 
� "2�/+����������������������������
E__inference_transconvB_layer_call_and_return_conditional_losses_18277���I�F
?�<
:�7
inputs+���������������������������@
� "?�<
5�2
0+��������������������������� 
� �
*__inference_transconvB_layer_call_fn_18287���I�F
?�<
:�7
inputs+���������������������������@
� "2�/+��������������������������� �
E__inference_transconvC_layer_call_and_return_conditional_losses_18233���J�G
@�=
;�8
inputs,����������������������������
� "?�<
5�2
0+���������������������������@
� �
*__inference_transconvC_layer_call_fn_18243���J�G
@�=
;�8
inputs,����������������������������
� "2�/+���������������������������@�
E__inference_transconvE_layer_call_and_return_conditional_losses_18189�}~J�G
@�=
;�8
inputs,����������������������������
� "@�=
6�3
0,����������������������������
� �
*__inference_transconvE_layer_call_fn_18199�}~J�G
@�=
;�8
inputs,����������������������������
� "3�0,����������������������������