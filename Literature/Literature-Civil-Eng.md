Received: 9 June 2025 Accepted: 26 October 2025

DOI: 10.1111/mice.

**INDUSTRIAL APPLICATION**

# Enhanced precision in axle configuration inference for

# bridge weigh-in-motion systems using computer vision and

# deep learning

## Domen Šoberl^1 Jan Kalin^2 Andrej Anžlin^2 Maja Kreslin^2

## Klen Čopič Pucihar^1 ,3 Matjaž Kljun1,3 Doron Hekič^2 Aleš Žnidarič^2

(^1) Faculty of Mathematics, Natural Sciences
and Information Technologies, University
of Primorska, Koper, Slovenia
(^2) Slovenian National Building and Civil
Engineering Institute, Ljubljana, Slovenia
(^3) Department of Information Science,
Stellenbosch University, Stellenbosch,
South Africa
**Correspondence**
Domen Šoberl, Faculty of Mathematics,
Natural Sciences and Information
Technologies, University of Primorska,
Glagoljaška 8, 6000 Koper, Slovenia.
Email:domen.soberl@famnit.upr.si
**Funding information**
The Slovenian Research and Innovation
Agency, Grant/Award Numbers: P5-0433,
IO-0035, J5-50155, J7-50096; Research
Programs CogniCom, Grant/Award
Number: 0013103; University of
Primorska and the Building Structures
and Materials, Grant/Award Number:
P2-0273; Slovenian National Building and
Civil Engineering Institute
**Abstract**
Heavy goods vehicles (HGVs) have a significant impact on road and bridge
infrastructure, with overloaded vehicles accelerating structural deterioration
and increasing safety risks. Bridge weigh-in-motion (B-WIM) systems estimate
gross vehicle weight (GVW) using strain measurements, but inaccuracies in axle
configuration recognition can reduce reliability. This study presents a low-cost
computer vision (CV) extension for existing B-WIM installations that verifies
strain-inferred axle configurations using traffic camera images and flags GVW
estimates as reliable or unreliable. Experiments on a data set of over 30,
HGVrecordsshowthatbycombiningconvolutionalneuralnetworkswithstrain-
based heuristics, GVW reliability can improve from 96.7% to 99.89%, effectively
excluding nearly all erroneous measurements. The approach operates without
interrupting ongoing B-WIM operations and can be applied retrospectively to
historical data. Limitations include the inability to detect raised axles (RAs),
which the method excludes as unreliable. This method provides a practical,
high-precision enhancement for structural health monitoring of bridges.

## 1 INTRODUCTION

Early applications of deep learning (DL) in civil engineer-
ing date back nearly a decade. Examples include the use of
deep restricted Boltzmann machine (RBM) models to esti-
mate concrete properties from mixture proportions (Rafiei
et al.,2017 ) and the application of advanced machine
learning models for construction cost estimation that

This is an open access article under the terms of theCreative Commons Attribution-NonCommercial-NoDerivsLicense, which permits use and distribution in any medium,
provided the original work is properly cited, the use is non-commercial and no modifications or adaptations are made.
© 2025 The Author(s). _Computer-Aided Civil and Infrastructure Engineering_ published by Wiley Periodicals LLC on behalf of Editor.

```
incorporate physical and financial project factors, as well
as economic variables and indexes (Rafiei & Adeli,2018).
More recently, DL techniques have been applied to vehi-
cle load identification on bridges (Chang et al.,2025 )and
to improve the accuracy of vehicle weight estimation (Yan
et al.,2025 ), which is crucial for detecting overloaded
trucks and assessing bridge damage caused, in particular,
by heavy goods vehicles (HGVs).
```
_Comput Aided Civ Inf._ 2025;40:6201–6216. wileyonlinelibrary.com/journal/mice **6201**


**6202** ŠOBERL et al.

HGVs, as defined by the EU, are motor vehicles with at
least four wheels with or without a trailer (i.e., road trac-
tors, road tractors with semi-trailers, or lorries), with a per-
missiblegrossvehicleweight(GVW)ofover3.5tonnesand
are solely used for transporting goods (Slootmans,2023 ).
These vehicles can significantly accelerate the deteriora-
tion of roads and, to a lesser extent, bridges, particularly
when fully loaded. Their size and weight exert signifi-
cant stress on road surfaces and bridge structures, which
reduces their lifespan, poses a safety risk to travelers, con-
tributes to vehicle wear and tear, and reduces driving
comfort (Meiring & Myburgh,2015 ). Overloaded HGVs are
especiallyproblematic,compromisingvehiclestabilityand
safety, increasing the likelihood of road accidents (Shah
etal.,2016 ).Furthermore,underpowered,overloadedvehi-
cles cause longer overtaking times, lower uphill speeds,
andhigherfuelconsumption(Kirushnath&Kabaso,2018),
while they also significantly accelerate damage to road
infrastructure (Pais et al.,2013 ; Rys et al.,2016 ).
AnestablishedmethodforcollectingHGVloadinginfor-
mation and enforcing weight regulations is the use of
weigh-in-motion (WIM) systems (Jacob & Feypell-de La
Beaumelle,2010; Sujon & Dai,2021 ). These systems can
record GVWs, axle loads, axle distances, and other impor-
tant information about HGVs in a particular road section.
Although WIM measurements are less accurate than tra-
ditional static weighing on dedicated platforms under
controlled conditions, they are significantly faster than
other weighing methods and do not require stopping the
traffic.WIMistheonlytechnologythatcapturesloadinfor-
mation for the entire traffic flow on a specific road section.
Bridge WIM (B-WIM) (Moses,1979 ) systems instrument
existing bridges or culverts as scales to weigh the crossing
vehicles. They do not require road closures during instal-
lation or maintenance and are more durable because no
system components are in direct contact with the vehicles.
The weighing challenge of a traditional B-WIM system
istominimizethedifferencebetweenthemeasuredstrains
and the strains calculated using the system of equations:

### 𝑠(𝑡𝑗)=

### ∑𝑁

```
𝑖=
```
### 𝐴𝑖𝐼

### [

### 𝑣𝑖

### (

### 𝑡𝑗−𝑡𝑖

### )]

### ;𝑗=1...𝑀 (1)

where𝑠(𝑡𝑗)are the calculated strains at a time𝑡𝑗,caused
by a vehicle,𝐴𝑖are the unknown axle weights of this vehi-
cle,𝐼(𝑥)represents the known influence line,𝑣𝑖denote the
axle velocities,𝑡𝑖are the times at which individual axles
arrive,𝑁refers to the total number of axles, and𝑀indi-
cates the total number of strain measurements (Moses,
1979 ). An influence line describes the variation of a func-
tion, such as shear, bending moment, or strain, under the
moving unit load. Figure1 illustrates how the bending
moment, or the proportional strain of a bridge at time𝑡𝑗,
is calculated using the known influence line and𝑁axle

```
FIGURE 1 Calculation of theoretical bridge response with
Equation (1), from Žnidarič et al. (2018).
```
```
weights that are not known in advance, in this case, from
𝐴 1 to𝐴5(Žnidarič et al.,2018). The influence line must
be derived from measurements on the particular bridge to
provide accurate results.
Accurate axle recognition is essential for precise vehicle
weighing.Initially,axlesweredetectedusingsensorseither
permanently embedded in the road’s wearing course or
temporarily mounted on its surface. The latter method was
mainly used for mobile installations but had a short lifes-
panduetodirectwheelimpacts.Overthepasttwodecades,
surface-mounted axle detectors have largely been replaced
bythefree-of-axle-detector(FAD)approach,whichderives
axle positions and spacings from additional strain sen-
sors installed on the underside of the bridge. This method
offers a significant advantage over other WIM systems in
eliminating the need for traffic closures during installation
and maintenance. However, it provides less reliable axle
information, as the data are acquired indirectly by sensors
located on the underside of the superstructure, rather than
in direct contact with the wheels.
Due to load transfer through the bridge superstructure,
the signal peaks representing individual axles are smeared
and often cannot be accurately located. As a result, axles
may be misidentified, with some being missed and others
incorrectly added. This problem has been mitigated using
complex signal-processing algorithms (Žnidarič et al.,
2018 ). As bridge length and traffic density increase, the
system of equations for calculating axle loads gradually
becomes ill-conditioned. One way to address this issue
is by grouping closely spaced double or triple axle con-
figurations to reduce the number of equations. The axle
configuration of the truck in Figure1 consists of three
groups of axles: two groups with one axle each and one
group with three axles, which can be denoted as a 113 con-
figuration. If the axle configuration is correctly inferred
from the strain data, the peaks in the strain signal can be
used to estimate the GVW of the passing HGV, since the
weight associated with a peak is distributed among the
axles in the same group.
```

ŠOBERL et al. **6203**

Currently, there is no reliable estimate of the accuracy
of the above method for inferring axle configurations from
strain data in a real-world environment. There is also no
method for estimating the reliability of individual GVW
estimations from strain data. Such reliability assessment
could help eliminate potentially unreliable outputs and
improve the overall precision of the calculated GVWs,
resulting in higher output credibility for B-WIM. In this
paper, we propose a low-cost solution that does not require
interrupting an operational B-WIM system. This solu-
tion double-checks the strain-inferred axle configurations
using a traffic camera installed at the B-WIM site and
flags each GVW as either reliable or unreliable. We show
that the credibility of GVW estimation can be improved
significantly by adding the proposed solution to an exist-
ing B-WIM site. We conducted experiments on a data set
collected at a B-WIM site on a highway near Ljubljana,
Slovenia, in 2014 and 2015, which contains over 30,
HGV records.
The assumptions of this paper are as follows:

1. The proposed solution should be low-cost and involve
    little or no interruption of the existing B-WIM instal-
    lation. It should be applicable retrospectively to data
    collected in the past, provided that a traffic camera was
    used at the site in question.
2. The accuracy of the GVW estimate depends on the cor-
    rectness of axle configuration inference. We assume
    that correct axle group inference yields a reliable GVW
    estimate.
3. We are interested in estimating the weight of heavier
    vehicles. Discarding the GVWs of lighter or unloaded
    trucks is not considered a significant drawback.

```
The contributions of this paper are as follows:
```
(1) the assessment of the accuracy of existing B-WIM axle
recognition heuristics;
(2) adaptation and assessment of existing DL architec-
tures for recognizing axle configurations from low-
resolution traffic camera photos;
(3) integrationoftheproposedDLmethodintoanexisting
B-WIM installation to improve the credibility of GVW
estimates.

The remainder of the paper is organized as follows: Sec-
tion2 provides an overview of related work in the field
of WIM axle recognition. Section3 defines the problem
andformulatestheresearchquestions.Themaincontribu-
tion is presented in Section4, where the proposed methods
are described. The application of these methods to a spe-
cific data set is discussed in Section5, and the results are
presented in Section6. Finally, key findings and limita-

```
tions are discussed in Section7, and the paper concludes
in Section8.
```
## 2 RELATED WORK

```
B-WIM systems calculate the axle loads and GVW of HGVs
crossing the bridge based on bridge deformation, which
is typically measured with strain transducers. Except in
cases where this is not possible (such as excessively long
spans or very high superstructures), the strain transduc-
ers also provide information used to detect axles (Moses,
1979 ; Žnidarič & Kalin,2020 ). FAD systems are inher-
ently more robust, but the axle information they provide
is less accurate than that from traditional wheel-triggered
axle detectors, particularly at longer spans and with
higher superstructures.
Recent advances in computer vision (CV) have enabled
the development of alternative methods for determining
vehicle parameters using traffic cameras. CV-based WIM
canbebroadlycategorizedintotwogroups:(1)non-contact
WIM (NC-WIM) or visual WIM (V-WIM), which uses only
a camera, and (2) hybrid systems that combine CV with
strain data to improve weight estimation accuracy.
NC-WIM methods aim to determine all necessary
weighing parameters by applying CV techniques to tire
images captured by a roadside camera. The tire contact
area is estimated by measuring the tire contact length
along the road and the vertical tire deflection. Optical
character recognition (OCR) is also used to obtain tire
manufacturer information. The weight is then calculated
as the sum of the individual tire–road contact forces (Feng
et al.,2020 ). Feng and Leung (2021 ) initially assumed sim-
ple shapes for contact patches, such as rectangles or ovals.
Kong et al. (2022 ) improved this by developing theoret-
ical equations for tire contact forces and simulating the
deformation process of different tire types in 3D. Zhang
et al. (2023 ) proposed a tire deformation detection method
based on CV and DL techniques. They created a data set of
tire images from different types of vehicles and proposed a
tire deformation calculation algorithm based on subpixel-
level edge detection, keypoint positioning, and scale factor
determination. They recently evaluated their approach in
a real-world toll station setting with 894 HGVs (Zhang,
OBrien, et al.,2024 ).
Zhang, Zhu, et al. (2024 ) argued that existing CV
methods for extracting tire deformations are too compu-
tationally intensive and time-consuming. They proposed
using the YOLOv5 model to segment tires and extract key
regions of interest (ROIs), such as the contact line, wheel
rim, and text on the tire sidewall. This method achieved
up to 17 times faster extraction of the tire–road contact
line compared to the method proposed by Feng and Leung
```

**6204** ŠOBERL et al.

(2021 ). Almutairi et al. (2022 ) reported achieving above
97% precision in recognizing individual tires using vari-
ous versions of YOLO. Gao et al. (2024 ) used a thermal
imaging camera to extract tire deformation under different
lightingandweatherconditions.Theycombinedathermal
image–based edge detection algorithm to detect tire defor-
mation and a YOLOv5-based sidewall marker detection
to obtain tire information. Feng et al. (2024 ) attempted a
similarthermalcameraapproachandalsoappliedexplain-
able machine learning methods. They found that the
most informative feature for training a vehicle weight
regression model is the ratio of tire–ground contact
length.
Recognizing individual tires, even with high accuracy,
does not guarantee correct axle configuration inference,
which depends on a carefully calibrated transformation
function that converts pixel measurements into spatial
coordinates (Xu et al.,2023 ;Yangetal.,2024 ). Instead, axle
types can be inferred directly through DL. The truck’s tax-
onomy, recognized from a photo, could serve as a lookup
reference in a database of trucks. Almutairi et al. (2022 )
reported 96% test accuracy when classifying 13 classes of
trucks. However, this does not detect the actual axles.
Wang et al. (2024 ) used an unmanned aerial vehicle
(UAV) with a high-resolution camera to detect vehicle axle
type, reporting 97.1% average precision. However, using
drones may be feasible for specific use cases and is likely
too difficult and costly for use at a commercial B-WIM
site.
We propose a low-cost DL approach that extends exist-
ing B-WIM sites without interrupting ongoing operations
and can also be applied to legacy B-WIM data. The model
infers axle configurations directly from images captured
by traffic monitoring cameras. Instead of relying solely
on the classifier’s output, we integrate it with existing
B-WIM heuristics to identify and filter out incorrectly
estimated GVWs.

## 3 PROBLEM DEFINITION

B-WIM systems currently rely on manually fine-tuned
heuristics to infer axle configurations from strain signals.
Figure2 shows an actual event at a B-WIM site, where
sensors captured a passing HGV. The sensors installed
beneath the bridge measured displacements along the
influence lines. The signals are processed through multi-
ple filters, each fine-tuned for the specific site, which are
depicted as multiple lines on the plot in the figure. A set of
heuristic rules, determined by experts, is used to infer the
positions of the truck’s axles, which are denoted by vertical
lines on the plot. Finally, based on predetermined thresh-
olds, axles that are close together are grouped. The axle

```
FIGURE 2 Strain signals (right) measured when an HGV (left)
crossed the B-WIM measuring site. The vertical bars denote
heuristically determined weights of individual axles.
```
```
configuration seen in Figure2 is depicted as 1211, which
means that the second and third axles are grouped, while
all other axles are single. The plots in the figure also show
that the second wave of signals is higher than the rest,
because more weight was applied by this group of axles on
the sensors. Information that the second wave belongs to
a group of two axles, rather than a single axle, is therefore
crucial to correctly interpret the signals.
If the B-WIM system determines that the vehicle is not
an HGV (based on the distance between the axles), the
event is discarded and the vehicle’s weight is not esti-
mated. In this way, all cars and other smaller vehicles are
ignored. The system also ignores the event if the presence
of multiple vehicles is recognized from the strain data.
A low-resolution traffic monitoring camera is typically
installed at a B-WIM site to capture a photo of the HGV,
which is then processed by the system. The position and
angle of the camera, as well as the quality of the captured
photos, are not necessarily ideal for automated recognition
of the vehicle’s axles. However, it should still be possible
to determine the vehicle’s axle configuration by eye. As
seen in Figure2, the visual conditions at the site used in
thisresearcharefarfromperfect;therefore,theresultspre-
sented in this paper should be achievable at other B-WIM
sites with similar or better conditions.
We propose a CV-based axle group detection method
that classifies traffic camera images into predefined
classes, each representing an axle configuration such as
113, 1211, and so on. We use machine learning to avoid
manualfine-tuningofparametersforspecificsites,thereby
eliminating tedious and error-prone expert intervention.
We run the proposed method in parallel with the exist-
ing B-WIM system, as shown in Figure3.Bothsystems
independently infer the axle configuration, and if they
agree on the output class, the estimated GVW is flagged
as reliable. The goal is to reduce the number of incor-
rectly estimated GVWs by discarding those flagged as
unreliable, which, in machine learning terminology, cor-
responds to the performance metric known as precision ,
wherecorrectlyestimatedreliableinstancesareconsidered
```

ŠOBERL et al. **6205**

**FIGURE 3** The proposed improved B-WIM axle configuration recognition method.

_true positives_ and incorrectly estimated reliable instances
areconsidered _falsepositives_ .Thisdiffersfromusingamul-
timodal approach, where both sources are input to the
samemodel.Keepingthemodelsseparatelowersthepossi-
bility of correlated noise and allows for more conservative
testing by cross-validating both outputs.
We identify two significant limitations to the proposed
approach in advance. First, the traffic monitoring cameras
cannot capture useful images at night or in poor weather
conditions, such as fog or heavy rain. Vehicles passing
the site under these conditions will not be considered for
GVW estimation by the proposed method, as they will
automatically be flagged as unreliable. However, the sig-
nificant drop in HGV traffic during nighttime mitigates
this problem.
The second limitation concerns raised axles (RAs).
When an HGV is lightly loaded or empty, drivers lift cer-
tain axles to reduce fuel consumption and tire wear. While
strain sensors will not register an RA, the camera will
detect it. This discrepancy between the two axle config-
urations would flag the result as unreliable, even if the
strain-based computations are correct. However, since we
are primarily concerned with measuring the weight of
heavily loaded vehicles, the exclusion of HGVs with RAs
does not necessarily represent a significant drawback.
We formulate the following research questions:

1. With what accuracy can axle configurations of HGVs be
    recognized using DL methods?
2. How much data are needed to train a DL model for axle
configuration recognition?
3. How much can the precision of strain-based B-WIM
axle configuration recognition heuristics be improved
by incorporating a traffic camera and DL methods?

## 4 METHODS

```
This section describes the methods used in the proposed
extension to the existing B-WIM system, as shown in
Figure3.WhentheB-WIMsysteminfersanaxleconfigura-
tion from the strain data and the heuristic rules determine
that the strain event is caused by an HGV, the CV mod-
ule is activated. If the axle configuration falls within the
classification scope of the CV module (some very rare
axle configurations may not be detectable through our CV
method), the photo taken at the moment of the strain
event is processed through a three-stage pipeline: vehi-
cle segmentation, image preprocessing, and axle group
classification, which is performed using a pretrained con-
volutional neural network (CNN). If the CNN recognizes
the same axle configuration as B-WIM, the instance is
flagged as reliable; otherwise, it is flagged as unreliable.
```

**6206** ŠOBERL et al.

**FIGURE 4** The process of generating training instances from the raw camera feed.

## 4.1 Vehicle segmentation

The vehicle segmentation stage involves identifying the
typeofvehicleandcomputingitsboundingboxes.Thiscan
be accomplished using the YOLO (You Only Look Once)
object detection and segmentation model (Redmon et al.,
2016 ). We used the pretrained YOLOv8 model, which can
detect 80 different object classes, including various types
of motorized road vehicles such as cars, trucks, buses, and
motorcycles. In our case, no additional training was neces-
sary, as the pretrained weights performed well on our data.
We considered only segments tagged as “truck” or “bus”
and ignored the rest.
Although the B-WIM system recognizes and discards
events with multiple vehicles present, YOLO will detect
other vehicles outside the range of the strain sensors, typ-
ically HGVs far behind the measured one or those on
the opposite side of the highway, driving in the oppo-
site direction. This issue can be addressed easily, since
the measured vehicle is closest to the camera and there-
fore occupies the largest area in the photo. In cases with
multiple HGVs in the photo, the segment covering the
largest area is selected. An example is shown in Figure4,
where the camera captured two HGVs driving one after
another. The first was correctly identified as the one being
measured.
There are rare cases where YOLO fails to correctly
detect or segment the measured HGV. Most often, YOLO
segments only the tractor instead of the entire tractor-and-
trailer composition, effectively splitting the truck in half.
As a result, only the front part of the HGV is processed for
axle detection, leading to a class with fewer axles. In our
experiments,about1%ofimageswereincorrectlyclassified
due to YOLO errors. These cases are flagged as unreliable
because of a mismatch between the CV and strain-based
inference of axle configurations.
During nighttime or under very bad weather conditions
(such as thick fog), YOLO fails to recognize and segment
the HGV entirely. In such cases, the event is discarded and
not considered for GVW estimation.

## 4.2 Image preprocessing

```
The traffic camera used with our B-WIM system captured
images at a640 × 480pixel resolution in RGB (24-bit) color
format. After cropping the segmented vehicle, the images
used for classification are small, rarely exceeding 200–
pixels in width or height. However, many popular CNN
models require RGB images with a resolution of224 × 224
pixels as input, which is close to our original image res-
olution. This usually requires only slight resizing of the
cropped part to fit the required224 × 224format.
To tackle overfitting during CNN training, image pre-
processing may include slight distortions, such as random
scaling and horizontal or vertical shifting. We apply these
modifications separately to each batch of training sam-
ples, so the samples differ slightly in each training epoch.
However, we limited these modifications to avoid losing
too much visual information through scaling or shifting
an axle beyond the visible edge of the image. The images
were scaled by a random factor between 0.9 and 1.0, then
randomly shifted horizontally and vertically within the
borders of the original224 × 224pixel area. We did not use
image rotation, as the traffic camera is mounted at a fixed
orientation. Rotation would train the CNN for viewing
angles that are irrelevant for inference.
AnexampleofimagepreprocessingisshowninFigure4.
First, YOLO is used to detect and segment an HGV (a). If
multiple HGVs are recognized (the two rectangles in the
second image), the one with the largest is selected. The
image is then cropped and scaled to match the224 × 224
format (b). Finally, random scaling and translations are
applied to introduce variation into the training process (c).
```
## 4.3 CNN classifiers

```
We trained and evaluated five popular classification mod-
els: VGG16 and VGG19 (Simonyan & Zisserman,2015 ),
DenseNet (Huang et al.,2017 ), MobileNet (Howard et al.,
2019 ), and ResNet (He et al.,2016 ). The selection was based
```

ŠOBERL et al. **6207**

on two factors: (1) each architecture accepts input images
of224 × 224pixels with three color channels (RGB), a
widely adopted format compatible with popular image
data sets, and (2) a pretrained implementation is available
in the TensorFlow/Keras library (Abadi et al.,2016 ). We
therefore focus on established, widely adopted industry-
standard CNN architectures that are publicly accessible
and available as pretrained models.
The chosen classification models are based on CNNs.
A CNN is a DL architecture specifically designed to pro-
cess data with a grid-like structure, such as images. It
consists of multiple convolutional layers that apply learn-
able filters to extract local features, pooling layers that
reducespatialdimensionalitywhileretainingkeyinforma-
tion, and fully connected layers that learn to translate the
extracted image features into output classes. The popular
choice of a224 × 224input format is a reasonable trade-
off between the complexity of the CNN architecture and
sufficiently distinctive visual information, while the input
dimension 224 can be divided by 2 several times within the
architecture.
We accommodate the size of the final layers to match
the number of distinct axle configurations our system can
recognize. After experimentation, we found that adding
a dense layer with 512 neurons after the convolutional
layers, followed by a drop out layer with a rate of 0.
to improve generalization, and an output layer with the
number of distinct classes, works best for this domain.
This is a notable reduction from their original sizes, there-
fore, the number of trainable parameters listed in this
paper for each architecture differs from the numbers in the
original publications.
All five CNN architectures are available as pretrained
models, capable of recognizing 1000 different object cat-
egories, such as “balloon,” “castle,” “necklace,” from the
ImageNet(Dengetal.,2009)database.Thesecategoriesare
not relevant to our domain, so additional learning (trans-
fer learning) is needed to adapt the network to our axles
domain. However, the pretrained weights in the initial
convolutional stages, which learn low-level image fea-
tures, might accelerate the training process. We compare
the speeds of the five models when training from scratch
versus using transfer learning.

## 4.4 Training the models

The models are trained on a large set of manually labeled
photos of HGVs. Each photo was examined by an expert
who determined its true axle configuration based on their
knowledge of B-WIM heuristics, primarily the thresholds
for axle groups. Very rare axle configurations may be
ignored, meaning these will automatically be flagged as

```
unreliable once the system is deployed. In our data set, we
observed a few unusual axle configurations that appeared
onlyafewtimesover severalyears.Ignoringthesedoesnot
pose a significant disadvantage in practice.
We considered introducing an additional class called
other , to which unknown axle configurations would be
assigned. However, our experiments showed an overall
decrease in classification accuracy (CA) when using the
other class, indicating that the neural network failed to
learn the abstract meaning of this category. Many known
instances that were previously classified correctly were
predicted as other when this class was introduced. More-
over, even if a predictor correctly learns to use such
a category, it cannot be compared with the B-WIM’s
prediction, so the GVW estimate would be flagged as
unreliable anyway.
After acquiring the labeled data set, classes with suf-
ficient representation for training, as well as their order,
aredetermined;forexample,𝐶 0 =113,𝐶 1 =1211,𝐶 2 =122.
One-hot encoding is used to encode the ground truth,
assigning the value 1.0 to the true class𝐶𝑘,and0toall
other classes. The class label𝐶𝑘is thus represented as
[𝐶 0 =0,𝐶 1 =0,...,𝐶𝑘=1,...,𝐶𝑛=0]within the training
set. During inference, the output vector represents a prob-
ability distribution over the classes, and the instance is
classified into the class with the highest probability.
To train the neural network, we used the categorical
cross-entropy loss function, defined as:
```
### 𝐿(𝑦,𝑦) = −̂

### ∑𝑛

```
𝑖=
```
```
𝑦𝑖log(𝑦̂𝑖) (2)
```
```
where𝑦is the one-hot encoded ground truth and𝑦̂is
the output class probability distribution. To minimize
the loss during training, we used the Adam optimiza-
tion method (Kingma & Ba,2015 ) with the learning rate
set to𝛼 = 0.0001, which is a typical choice for this type
of optimization problem and performed well with our
data set.
```
## 4.5 Evaluation

```
To evaluate the improvement in B-WIM precision, we use
the evaluation metric shown in Figure5. The baseline per-
formance is the CA of the strain-based B-WIM heuristics.
With the ground truthavailable, we can determine the per-
centage of correctly inferred axle configurations, denoted
as C in the figure, while incorrect inferences are denoted
as I.
Introducing the proposed CV-based extension, referred
to here as the discriminator , the strain-based inferences
are flagged as either reliable (positive) or unreliable
```

**6208** ŠOBERL et al.

**FIGURE 5** Measuring the performance of the proposed
B-WIM system.

(negative). This results in four types of events that align
with the machine learning concepts of _true positive_ (TP),
_false positive_ (FP), _false negative_ (FN), and _true negative_
(TN) predictions. For example, if the strain-based system
correctly infers the axle configuration and the discrimina-
tor agrees with its output, the event is considered a _true
positive_. If the discriminator disagrees with that correct
output, it is a _false negative_ event, and so on. We measure
_precision_ and _recall_ , defined as:

```
precision= 𝑇𝑃
𝑇𝑃 + 𝐹𝑃
```
### (3)

```
recall=𝑇𝑃 + 𝐹𝑁𝑇𝑃 (4)
```
High precision means that the number of incorrect axle
configurations within the _reliable_ set is low, allowing us
to have a high level of confidence in the correctness of
the computed GVWs. High recall means that our process
of elimination did not exclude many correct instances.
Note that by removing the discriminator, we have FN=
0 and TN=0, so the CA of the baseline strain-based
heuristic equals its precision, while its recall is 100%.
Therefore, we aim to improve the precision of the base-
line system while trying to keep the reduction in recall
minimal.

## 5 APPLICATION

The proposed methods were applied to a data set collected
at a highway near Ljubljana, Slovenia, between 2014 and

2015. The B-WIM system installed under an 8-m-long slab
bridgerecordeditsresponsesduringtheHGVscrossingthe
bridge, while a traffic camera mounted on the bridge cap-
tured still images of the traffic whenever an HGV triggered
the predefined strain levels.

```
TABLE 1 Distribution of the available data samples by classes.
```
```
Class
```
```
Labeled samples Validation set
Number Percentage Number Percentage
113 11,921 38.98% 524 47.51%
1211 4550 14.88% 25 2.27%
122 3705 12.12% 56 5.08%
11 2764 9.04% 340 30.83%
22 2440 7.98% 7 0.63%
111 1611 5.27% 9 0.82%
112 1516 4.96% 105 9.52%
1112 669 2.19% 4 0.36%
12 460 1.50% 24 2.18%
1111 331 1.08% 3 0.27%
123 267 0.87% 4 0.36%
1212 219 0.72% 1 0.09%
1222 128 0.42% 1 0.09%
Unused 309 1.01% 5 0.45%
```
## 5.1 Data labeling

```
The experts manually examined 30,890 photos of HGVs
and determined their true axle configurations. They
identified 44 different classes, but most were heavily
underrepresented—11 contained only one vehicle and 23
contained fewer than 10. We included only classes with at
least 100 vehicles to ensure some sample variety, finally
resulting in 13 classes and 30,581 training instances. Their
distribution is shown in Table1.
The data set is heavily imbalanced, with most vehi-
cles belonging to class 113—the 2-axle tractor with a
3-axle semi-trailer. However, this issue is addressed dur-
ing training by upsampling the classes using the image
preprocessing method shown in Figure4.
Note that the distribution of the labeled data does not
necessarily represent the real-world distribution of vehicle
types, as the labeling was done arbitrarily. A more accurate
approximation of the actual distribution can be obtained
by analyzing data collected over a specific period, which
we have done when selecting the validation set.
```
## 5.2 Validation set

```
As the validation set, the day with the highest count of
labeled instances was chosen, which was March 10, 2014.
There were 1691 events recorded on that date, of which
YOLOrecognizedanHGVin1256cases,failingmostlydur-
ing nighttime. The distribution of events on that day by
hour is shown in Figure6.
There were 15 cases where YOLO recognized an HGV
but failed to segment it correctly. The error was either
```

ŠOBERL et al. **6209**

**FIGURE 6** The hourly distribution of HGVs on the validation
day: Detected by the existing B-WIM strain-based system (dotted),
segmented by YOLO (dashed), and labeled by our experts (solid).

cutting off the tractor from the trailer or missing the
frontal part of the vehicle in the photo, making correct
segmentation impossible. The latter case was due either
to incorrect camera timing or the vehicle driving above
the presumed speed. There were 133 cases where YOLO
correctly segmented the HGV, but our experts could not
determine the ground truth, either because of poor visibil-
ity or partial axle occlusions. Most of these cases occurred
at night, when traffic headlights allowed YOLO to detect
and segment an HGV, but the axles were not sufficiently
illuminated for the expert to identify the correct class. Five
vehicles had rare axle configurations not recognizable by
our trained models: 2222, 121, 21, 3, and 114. We were left
with 1103 labeled validation instances, whose distribution
byclassisshowninTable1.

## 6 RESULTS

We conduct four types of experiments:

1. _Sample efficiency_ to assess the minimum number of
    instances needed to successfully train the models.
2. _Tenfold cross-validation_ to assess the CA of the trained
CNN models independently of the B-WIM framework.
3. _A separate validation_ to assess the improvement in per-
formance of the proposed system over an actual 24-h
operational period.
4. _An ablation study_ to evaluate the impact of sample aug-
mentation on the final B-WIM performance and the
impact of using pretrained weights on the speed of
training.

All experiments were conducted on an NVIDIA RTX
4080 GPU with 16 GB of VRAM, using TensorFlow
machine learning library version 2.17.0.

```
FIGURE 7 The highest achieved classification accuracy (CA)
for different sizes of the training set.
```
## 6.1 Sample efficiency

```
Sample efficiency was evaluated by gradually increasing
the training set size and recording the highest CA achieved
by any of the five CNN architectures. We selected 10 sub-
sets of different sizes as they became available during the
lengthy manual data labeling process, using the follow-
ing approach. For a predefined maximum class size𝑆,we
randomly selected𝑆instances from each class. If a class
contained fewer than𝑆instances, all available instances
from that class were used. All remaining instances were
used for testing. No upsampling was performed for classes
with fewer than𝑆 samples, so sample efficiency was
measured only on unique samples.
We sampled 10 training subsets using𝑆=10, 20, 50, 100,
200, 500, 1000, 2000, and 5000. The CA for each train-
ing set is shown in Figure7.The x -axis represents the
number of samples used. The numbers next to the dots
indicate the values of𝑆. About 2000 unique instances are
sufficient to achieve over 90% CA, after which progress
slows significantly.
```
## 6.2 CNN performance

```
The stand-alone CNN performance was evaluated sepa-
rately for all five CNN architectures using stratified 10-fold
cross-validation. To avoid training a biased model, under-
represented classes were upsampled by duplicating exist-
ing samples until each class contained 5000 samples. The
5000 samples from the 113 class were selected randomly.
Sample augmentation was performed automatically on
each training batch of 32 samples by the image prepro-
cessing method presented in Figure4. Transfer learning
was conducted on models pretrained on the ImageNet
database. We did not apply the incremental unfreezing
```

**6210** ŠOBERL et al.

**FIGURE 8** Training progress when conduction transfer learning (left) and when training the models from scratch (right).

**TABLE 2** Classification accuracy (CA) and B-WIM performance of the five convolutional neural networks.

```
Base CNN
architecture
```
```
Trainable
parameters
(millions)
```
```
Batch
training time
```
```
Sample
classification
time
```
```
CA B-WIM performance
10-fold cross-
validation
```
```
24-hour
validation precision
```
```
recall recall
(incl. RA) (excl. RA)
VGG16 27.6 127 ms 45 ms 97.90% (1.91%) 98.01% 99.89% 87.45% 98.63%
VGG19 32.9 148 ms 48 ms 98.08% (1.20%) 97.55% 99.68% 86.61% 97.68%
ResNet101V2 93.9 108 ms 65 ms 96.74% (2.58%) 98.10% 99.79% 87.45% 98.52%
DenseNet121 32.7 94 ms 72 ms 97.47% (1.60%) 94.74% 99.56% 84.64% 95.26%
MobileNetV3 15.4 44 ms 57 ms 92.99% (5.18%) 92.29% 99.66% 83.33% 93.39%
```
technique,butinsteadunfrozealllayersatthebeginningof
training. We did not find it necessary to perform any such
fine-tuning, as training converged in just a few episodes, as
shown by the left plot in Figure8.
CA achieved by each of the five models is shown in
Table2.The _10-fold cross-validation_ column shows the
average CA over the 10 folds, with the standard deviation
given in parentheses. The _24-h validation_ column shows
theresultsachievedonthevalidationsetthatwasexcluded
from training. Note that the main difference between the
two evaluations is the class balance. In both cases, the
models were trained on a balanced (upsampled) data set,
but the validation data set is unbalanced and reflects the
actual class distribution on site. Both VGG-based models
showed the most reliable performance and resilience to
class distribution.
Table3 shows the cumulative confusion matrix for the
validation set, where the classifications from all five neu-
ral networks are summed. Thus, the matrix displays5⋅
1103 = 5515classifications. This allows us to examine the
most common errors made by convolutional classification
in general. There are three types of errors: missing an axle,

```
adding a nonexistent axle, or joining axles that should be
recognized as separate.
Lastly, we may consider the size of the models in rela-
tion to their accuracy. We observe that the number of
training parameters does not directly correlate with CA.
Larger models with more parameters do not necessar-
ily outperform smaller ones, indicating that performance
depends more on architectural design choices than on
model size. In this regard, the choice of VGG16 among
the five tested seems most reasonable, as its perfor-
mance is similar to VGG19, while its smaller size allows
for slightly faster training and inference. Moreover, our
experiments in either enlarging or shrinking the VGG-
based structure did not yield better results than those
presented.
There seems to be an interesting anomaly in Table2,
where MobileNet takes more time to infer a single instance
thantotrainonabatchof32instances.Thislikelyoccurred
due to poor GPU parallelism utilization when process-
ing a single instance. Running the same experiment on
a CPU resulted in training times about 30 times longer
than inference.
```

ŠOBERL et al. **6211**

**TABLE 3** Confusion matrix of the validation set, cumulative for all five convolutional neural networks. True values are in rows,
predictions are in columns. The most noticeable errors are shaded.

```
113 1211 122 11 22 111 112 1112 12 1111 123 1212 1222
11325891106114000800
1211 0 118 7 0 0 0 0 0 0 0 0 0 0
122 2 0 263 2 706000000
11 0 0 1 1664 4 10 4 0 17 0 0 0 0
2200003500000000
111 0 0 0 0 0 45 0 0 0 0 0 0 0
112 3 0 93115492002000
1112 1 0 1 0 1 0 0 17 0 0 0 0 0
120004100001060000
1111 0 0 0 0 3 0 0 0 0 12 0 0 0
12300201020001500
1212 0 1 0 0 0 0 0 0 0 0 0 4 0
12220000000000005
```
**TABLE 4** B-WIM performance on the validation set using
different CNN architectures (lower is better). ‘Incorrect GVW’ (bold
values) is the key metric for evaluating improvements over the
baseline method.

```
Method
```
```
Incorrect
axles
```
```
Incorrect
GVW
```
```
Lost
all
```
```
Lost
non-RA
VGG16 22 1 134 13
VGG19 27 3 143 22
ResNet101V2 21 2 134 14
DenseNet121 58 4 164 45
MobileNetV3 85 3 178 115
Baseline 35 35 0 0
```
## 6.3 The improved B-WIM performance

Measuring the baseline CA of the existing B-WIM strain-
based heuristicsisstraightforward.Ofthe30,890manually
labeled instances, the existing B-WIM heuristic correctly
inferred 29,550 instances, resulting in a 96.70% CA. In the
validation set, 1068 out of 1103 labeled instances were cor-
rectly inferred, which is a 96.83% CA. This means that the
GVW of 35 vehicles was certainly incorrectly estimated on
the validation day.
The _B-WIM performance_ columns in Table2 show how
much different CNN architectures improve the baseline
B-WIM precision of 96.70%, as well as how much the
recall decreases from the baseline of 100%, depending on
whethervehicleswithRAsareconsidered.Table4presents
the B-WIM performance on the validation set as specific
case counts in terms of losses, which means lower number
is better. The _incorrect axles_ column indicates how many
of the 1103 labeled cases were incorrectly identified by the
CNN alone. The _incorrect GVW_ value, which is the focus
of this research, shows how many of the “reliable” GVWs

```
were incorrectly inferred. The lost all columns indicate
how many GVW cases would have been correctly inferred
but were flagged as “unreliable.” The number of those
cases without an RA is shown in the last column. These
are potentially the heavier vehicles that may be of interest
for weighing.
Statistical analysis of precision on the validation set
is presented in Table5. A paired bootstrap significance
test with 5000 resampling iterations is used to compare
improvements in B-WIM precision when using different
CNN models. The table presents selected comparisons, as
showing all possible pairs would be too exhaustive.
The results indicate that using the CV-extension with
any of the five CNN models significantly outperforms
the baseline B-WIM performance. There is no signifi-
cant difference among the top three performing models:
VGG16, VGG19, or ResNet. DenseNet and MobileNet,
which showed notably lower CA during validation, also
show a statistically significant drop in B-WIM precision
(𝑝<.05) when used instead of the top three. However, the
evidence (𝑝 = .0388) is not very strong.
```
## 6.4 Ablation study

```
We study the effect of the pretrained weights on the
speed of training the models, and the effect of sample
augmentation shown in Figure4 on CA.
```
## 6.4.1 Transfer learning

```
There is a notable difference in training speed between
training models from scratch and performing transfer
learning on models pretrained on the ImageNet database.
However, both approaches achieve similar results after
```

**6212** ŠOBERL et al.

**TABLE 5** Results of the paired bootstrap significance test with 5000 resampling iterations. The table reports the statistical significance in
improving the B-WIM’s precision using different CNN models. The _baseline_ method uses no CV-based extension.

```
Compared methods
```
```
Mean
difference 𝒑
```
```
Significance 95% Confidence interval
( 𝒑<.𝟎𝟓 ) Lower Upper
VGG16 VGG19 0.2163 .1216 No 0.0000 0.
VGG16 ResNet101V2 0.1067 .5292 No −0.0007 0.
VGG19 ResNet101V2 0.1096 .4228 No −0.2146 0.
ResNet101V2 DenseNet121 0.2269 .0388 Weak 0.0026 0.
DenseNet121 MobileNetV3 0.3336 .0388 Weak 0.0030 0.
VGG16 Baseline 3.0662 <.001 Strong 2.0714 4.
VGG19 Baseline 2.8499 <.001 Strong 1.9039 3.
ResNet101V2 Baseline 2.9595 <.001 Strong 1.9688 4.
DenseNet121 Baseline 2.7326 <.001 Strong 1.7633 3.
MobileNetV3 Baseline 3.0662 <.001 Strong 2.0714 4.
```
**TABLE 6** Change in classification accuracy (CA) on the
validation set with and without sample augmentation, and its
statistical significance.

```
Base CNN
architecture
```
```
CA
augmented
```
```
CA not
augmented 𝝌𝟐 𝒑
VGG16 98.01% 94.47% 30.72 <.
VGG19 97.55% 93.74% 22.12 <.
ResNet101V2 98.10% 92.75% 50.21 <.
DenseNet121 94.74% 95.47% 0.98.
MobileNetV3 92.29% 86.94% 114.38 <.
```
a sufficient number of training epochs. Therefore, we
compareonlythedecreaseoflossvaluesovertrainingtime.
Transfer learning converged by the third training
episodeinallfivecases,asshownintheleftplotinFigure8.
In contrast, the training progress when training the mod-
els from scratch is shown in the right plot in Figure8.
The training progress is noticeably slower but eventu-
ally achieves the same results as the pretrained approach.
VGG-based architectures were the fastest, reaching results
similar to their pretrained counterparts after about 10
episodes. DenseNet and MobileNet caught up with their
pretrained counterparts in about 30 episodes. ResNet
showed the worst performance but managed to catch up
by episode 50. This can be attributed to its massive size.

## 6.4.2 Sample augmentation

Table6 shows the change in CA when upsampling under-
represented classes is done by duplicating existing images
without applying the scaling and translation described
in Section4.2. McNemar’s test was applied to assess
whether the difference in CA between the two upsampling
approaches was statistically significant.

```
FIGURE 9 Difficult, but correctly classified by all models.
```
```
As shown in the table, CA decreased by a few percent
when augmentation was not applied, except for DenseNet,
which did not show any significant change (𝑝 = .3222).
This provides strong evidence for the use of sample aug-
mentation,whichenablesbetterknowledgegeneralization
and does not have a detrimental effect on performance in
the worst case.
```
## 6.5 Case studies

```
We present several case studies of correct and incorrect
image classifications. Figure9 shows two challenging
examples that were correctly classified by all five CNN
models. Case (a) is barely recognizable to the naked
eye due to foggy conditions. However, the information
is present and discernible to the CNN. Case (b) shows
an HGV carrying other vehicles, and the photo includes
the tires of the loaded cars, which should not be consid-
ered part of the axle configuration. Numerous such cases
demonstrate that the neural network learned to correctly
focus on the HGV’s wheels while ignoring those of the
loaded cars.
Figure10 shows two trucks with RAs. Strain signals
do not register RAs, while the CNN includes them in the
```

ŠOBERL et al. **6213**

**FIGURE 10** Trucks with raised axles got flagged “unreliable.”

**FIGURE 11** Threshold error. In both cases, the last two axles
should not be grouped by CNN.

inferred axle configuration. Both systems are correct in
theirownwaybutdisagreeontheaxleconfiguration.Cases
(a) and (b) in the figure are recognized by the CNN as 12
and 113, respectively, while strain-based heuristics infer 11
and 112. Such cases are therefore flagged as “unreliable”
andexcludedfromGVWestimation.However,theycannot
be regarded as erroneous.
There were many cases where CNN classification did
not match the expert’s label, but upon closer inspection,
it becomes clear that the difference between the label and
the inferred class lies in the threshold that defines an
axle group. Typically, the CNN would group axles that
appear close together, but according to experts, they did
not meet the grouping threshold. This issue is largely due
to the sharp camera angle. With the two cases showed in
Figure11, CNN grouped the last two axles, while they were
labeled as separate.
There were also genuine CNN errors, which can be
attributed to poor model performance. The model would
either infer an axle where there is none (a “ghost axle”)
or miss an existing axle. Examples of both occurrences are
shown in Figure12. Typically, a shadow or part of the vehi-
cle’s body would be mistaken for an axle, or an axle would
be missed due to poor visibility or partial occlusion. The
latter case is largely due to the sharp camera angle.
Errors made by CV usually result in disagreement on
theinferredaxlesandflaggingtheinstanceas“unreliable.”
This represents either a true or a false negative instance,
so it does not negatively affect precision. Loss of preci-
sion occurs when both systems—the B-WIM’s baseline
heuristicsandtheproposedCVextension—makethesame

```
FIGURE 12 Computer vision error.
```
```
mistake. Although this seems highly unlikely, it happens
for one of the following two reasons:
```
1. B-WIM heuristics coincidentally infers a “ghost axle” at
    the same position where an actual axle is raised, and
    the CNN correctly recognizes it. Both systems therefore
    agree on the axle configuration, although this configu-
    ration is incorrectly inferred from strain signals. This is
    a methodological problem and is independent of CNN
    performance.
2. Both systems make the same axle grouping mistake
when the distance between two adjacent axles is close
to the grouping threshold. This problem is specific to
CNNs, since different models may learn different axle
grouping criteria.

## 7 DISCUSSION

## 7.1 Key findings

```
One of the most interesting findings of this research is that
popular architecture-based CNNs can learn to recognize
axle configurations from low-quality HGV photos with up
to 98% accuracy. Given the less-than-ideal angle of our
traffic camera—where even our experts were sometimes
unsure of the correct axle count and had to analyze the
strain signals—the results are quite impressive. This sug-
gests that the proposed CV-based classifier could improve
existingstrain-basedheuristics.However,itisimportantto
note that the precision achieved by either method alone is
lower than that of the proposed extended system.
Another notable result is the high precision of the pro-
posed system, which is nearly 100%. This high precision
is not entirely unexpected, given that for an error to
occur, both axle configuration classifiers must make the
same mistake. Since the B-WIM heuristics achieve approx-
imately 96% accuracy and the CNN classifier can achieve
about 98% accuracy, we might expect one such mistake
in about 1000 cases with the proposed method. This was
indeed observed when using the VGG16-based classifier
```

**6214** ŠOBERL et al.

on the validation set. However, the challenge of learning
a fine-tuned threshold for grouping axles under the given
camera angle led to a few more mistakes with other CNN
architectures. The severity of these mistakes is debatable,
as they are likely not as detrimental to the GVW computa-
tion as missing or adding an axle. A possible approach to
this issue could be fuzzy labeling, where the expert assigns
multiplelabelstoasingleinstancewithaprobabilityscore.

## 7.2 Limitations of the available data set

We tested our classifier only with photos taken in favorable
weather conditions and during daylight. In other words,
our experts labeled only those instances where axle con-
figurations were recognizable by the naked eye and could
therefore be labeled correctly. As a result, much of the traf-
fic recorded at night, in thick fog, or during heavy rain was
excluded from the experiment. While it could be argued
that high-quality training samples should be selected to
train a model, testing should ideally cover a broader range
of scenarios. However, since dark and heavily blurred pho-
toscouldnotbelabeledwithcompletecertainty,theycould
not be used to evaluate the accuracy of the trained clas-
sifiers. A possible alternative approach would be to apply
photo filters to simulate low-visibility conditions on the
labeled clear photos.
In a live scenario, we could either turn off the proposed
CV-based discriminator at night and revert to baseline per-
formance, or allow the discriminator to eliminate more
vehicles during nighttime, which should, in principle,
lower the recall rate but have minimal impact on the sys-
tem’s precision. Another option is to install night vision
or thermal cameras. However, in this research, we use
existing data acquired with current B-WIM installations.

## 7.3 Advantages and limitations of the

## method

A key limitation of our approach to axle configuration
recognition from photographs is its reliance on a predeter-
mined set of axle configuration classes. An axle configu-
ration not included in one of the training classes cannot
be verified and is flagged as unreliable a priori. However,
such configurations are very rare and are observed only a
few times per year, therefore, their exclusion from GVW
estimation represents a negligible loss.
On the other hand, recognizing axle configurations as a
whole, rather than identifying individual axles and group-
ing them afterward, offers two important advantages: (1)
It eliminates the need for error-prone manual fine-tuning
of axle grouping parameters for a specific site and (2) it

```
resolves the issue of car carriers, where the axles of car-
ried cars must be ignored. The first point is supported
by our results, in which a trained CNN alone exceeded
the accuracy of the existing B-WIM heuristics—manually
fine-tuned by engineers for several years—by a few per-
centage points. However, we recommend training the
models specifically for each chosen B-WIM site, as differ-
ent camera angles, distances, and other visual factors may
affect CA.
The problem of recognizing RAs remains a subject for
future work. The discrepancy between CV and sensor data
is a strong indicator of an RA, especially when considering
the typical positions of RAs within axle groups and inte-
grating this knowledge into the system. This should make
it possible to evaluate the probability of an RA. However,
errors are still possible, and including such cases in the set
of reliable GVW measurements would cause a certain drop
in precision.
Lastly, there is the question of generalization versus
memorization of the trained CNN models. Achieving
similar results on a separate validation set compared to
cross-validation on the training set may not guarantee gen-
eralization, since only a limited number of truck types
exist, and many may pass the same B-WIM site daily.
Therefore, excluding specific days from training may not
prevent the same trucks from appearing in the training
data, just recorded on different days. However, memoriza-
tion may not be a significant practical issue. Even if the
model learns to identify axle configurations by memoriz-
ing all existing types and brands of vehicles, this does not
affectthepracticalusefulnessofthesolution,exceptforthe
possibility that the models should be retrained every few
years as new types of vehicles appear on the roads.
```
## 8 CONCLUSION

```
This paper presents a low-cost CV extension for com-
mercially available B-WIM systems that can improve the
reliability of estimated GVWs from the existing 96.70%
to up to 99.89%. This reliability is quantified by the clas-
sification precision achieved by the proposed system, in
which unreliable GVW estimates are excluded. In practice,
this means that 34 out of 35 incorrectly measured vehicles
over a 24-h period were correctly identified as unreliable
and excluded from further analysis. The proposed solu-
tion can be integrated with existing B-WIM installations
without disruption or applied retrospectively to previously
collected B-WIM data.
The main limitation of the proposed method is the
inability of the CV models to recognize RAs under the typ-
ical visual conditions of traffic monitoring cameras. Our
compromise is to have the method exclude vehicles with
```

ŠOBERL et al. **6215**

RAs as unreliable, since these vehicles are usually lighter
and their GVW is less critical for monitoring structural
deterioration. Future work could address this problem by
investigating the use of multimodal learning (Erukude
et al.,2025 ) to combine strain signals with camera images,
or Dynamic Ensemble Learning (Alam et al.,2020 )to
combine several CNN models.

### ACKNOWLEDGMENTS

The authors would like to express their sincere gratitude to
theMotorwayCompanyoftheRepublicofSlovenia(DARS
d.d.) for their valuable support in this study. Special thanks
are also extended to CESTEL d.o.o. for their assistance
relatedtotheSiWIMsystem.Theauthorswouldliketorec-
ognizethecontributionsofNežaGošteandMirkoKosičfor
their support in building the ground truth data.
This research was funded by The Slovenian Research
and Innovation Agency, grant numbers P5-0433, IO-0035,
J5-50155 and J7-50096. This work has also been supported
by the research programs CogniCom (0013103) at the
University of Primorska and the Building Structures and
Materials (P2-0273) at Slovenian National Building and
Civil Engineering Institute.

### REFERENCES

Abadi, M., Barham, P., Chen, J., Chen, Z., Davis, A., Dean, J., Devin,
M.,Ghemawat,S.,Irving,G.,Isard,M.,Kudlur,M.,Levenberg,
J., Monga, R., Moore, S., Murray, D. G., Steiner, B., Tucker, P.,
Vasudevan, V., Warden, P., Wicke, M., et al. (2016). Tensorflow: A
system for large-scale machine learning. In _Proceedings of the 12th
USENIX conference on operating systems design and implementa-
tion_ (OSDI’16, pp. 265–283), USA. USENIX Association.https://dl.
acm.org/doi/10.5555/3026877.
Alam, K. M. R., Siddique, N., & Adeli, H. (2020). A dynamic
ensemblelearningalgorithmforneuralnetworks. _NeuralComput-
ing and Applications_ , _32_ (12), 8675–8690.https://doi.org/10.1007/
s00521-019-04359-
Almutairi, A., He, P., Rangarajan, A., & Ranka, S. (2022). Automated
truck taxonomy classification using deep convolutional neural
networks. _International Journal of Intelligent Transportation Sys-
tems Research_ , _20_ (2), 483–494.https://doi.org/10.1007/s13177-022-
00306-
Chang, L., Tang, Q., Xin, J., Jiang, Y., Zhang, H., Li, Z., Zhou, Y., &
Zhou, J. (2025). Low-complexity real-time detection Transformer
for identifying bridge vehicle loads. _Computer-Aided Civil and
Infrastructure Engineering_ , _40_ (26), 4485–4506.https://doi.org/10.
1111/mice.7 0061
Deng, J., Dong, W., Socher, R., Li, L.-J., Kai, L., & Li, F.-F. (2009).
ImageNet: A large-scale hierarchical image database. In _2009
IEEE conference on computer vision and pattern recognition_ (pp.
248–255), Miami, FL. IEEE.https://doi.org/10.1109/CVPR.2009.
5206848
Erukude, S. T., Veluru, S. R., & Marella, V. C. (2025). Multimodal
deep learning: A survey of models, fusion strategies, applica-

```
tions, and research challenges. International Journal of Computer
Applications , 187 (19), 1–7.https://doi.org/10.5120/ijca
Feng, J., Gao, K., Zhang, H., Zhao, W., Wu, G., & Zhu, Z. (2024). Non-
contactvehicleweightidentificationmethodbasedonexplainable
machine learning models and computer vision. Journal of Civil
Structural Health Monitoring , 14 (4), 843–860.https://doi.org/10.
1007/s13349-023-00757-
Feng, M. Q., & Leung, R. Y. (2021). Application of computer vision for
estimation of movingvehicle weight. IEEESensorsJournal , 21 (10),
11588–11597.https://doi.org/10.1109/JSEN.2020.
Feng, M. Q., Leung, R. Y., & Eckersley, C. M. (2020). Non-contact
vehicle weigh-in-motion using computer vision. Measurement ,
153 ,107415.https://doi.org/10.1016/j.measurement.2019.
Gao, K., Zhang, H., & Wu, G. (2024). A multispectral vision-based
machine learning framework for non-contact vehicle weigh-
in-motion. Measurement , 226 ,114162.https://doi.org/10.1016/j.
measurement.2024.
He, K., Zhang, X., Ren, S., & Sun, J. (2016). Identity mappings in
deepresidualnetworks.InLeibe,B.,Matas,J.,Sebe,N.,&Welling,
M., (Eds.), Computer vision—ECCV 2016 (Vol. 9908, pp. 630–645).
Springer International Publishing.https://doi.org/10.1007/978-3-
319-46493-0_
Howard,A.,Sandler,M.,Chen,B.,Wang,W.,Chen,L.-C.,Tan,M.,
Chu,G.,Vasudevan,V.,Zhu,Y.,Pang,R.,Adam,H.,&Le,Q.
(2019). Searching for MobileNetV3. In 2019 IEEE/CVF interna-
tional conference on computer vision (ICCV) (pp. 1314–1324), Seoul,
Korea (South). IEEE.https://doi.org/10.1109/ICCV.2019.
Huang, G., Liu, Z., Van Der Maaten, L., & Weinberger, K. Q. (2017).
Denselyconnectedconvolutionalnetworks.In 2017ieeeconference
oncomputervisionandpatternrecognition(CVPR) (pp.2261–2269),
Honolulu, HI. IEEE.https://doi.org/10.1109/CVPR.2017.
Jacob, B., & Feypell-de La Beaumelle, V. (2010). Improving truck
safety: Potential of weigh-in-motion technology. IATSS Research ,
34 (1), 9–15.https://doi.org/10.1016/j.iatssr.2010.06.
Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic opti-
mization. In Proceedings of the 3rd International Conference on
Learning Representations (ICLR 2015). https://arxiv.org/abs/1412.
6980
Kirushnath, S., & Kabaso, B. (2018). Weigh-in-motion using machine
learning and telematics. In 2018 2nd international conference on
telematics and future generation networks (TAFGEN) (pp. 115–120),
Kuching. IEEE.https://doi.org/10.1109/TAFGEN.2018.
Kong, X., Zhang, J., Wang, T., Deng, L., & Cai, C. (2022). Non-
contactvehicleweighingmethodbasedontire-roadcontactmodel
and computer vision techniques. Mechanical Systems and Sig-
nal Processing , 174 , 109093.https://doi.org/10.1016/j.ymssp.2022.
109093
Meiring, G., & Myburgh, H. (2015). A review of intelligent driv-
ing style analysis systems and related artificial intelligence
algorithms. Sensors , 15 (12), 30653–30682.https://doi.org/10.3390/
s
Moses, F. (1979). Weigh-in-motion system using instrumented
bridges. Transportation Engineering Journal of ASCE , 105 (3),
233–249.https://doi.org/10.1061/TPEJAN.
Pais, J. C., Amorim, S. I. R., & Minhoto, M. J. C. (2013). Impact
of traffic overload on road pavement performance. Journal of
Transportation Engineering , 139 (9), 873–879.https://doi.org/10.
1061/(ASCE)TE.1943-5436.
```

**6216** ŠOBERL et al.

Rafiei, M. H., & Adeli, H. (2018). Novel machine-learning model
for estimating construction costs considering economic variables
and indexes. _Journal of Construction Engineering and Manage-
ment_ , _144_ (12), 04018106.https://doi.org/10.1061/(ASCE)CO.1943-
7862.0001570,
Rafiei, M. H., Khushefati, W. H., Demirboga, R., & Adeli, H. (2017).
Supervised deep restricted Boltzmann machine for estimation of
concrete. _ACI Materials Journal_ , _114_ (2). https://doi.org/10.14359/
51689560
Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). You
Only Look Once: Unified, real-time object detection. In _2016
IEEEconferenceoncomputervisionandpatternrecognition(CVPR)_
(pp. 779–788), Las Vegas, NV, USA. IEEE.https://doi.org/10.1109/
CVPR.2016.
Rys, D., Judycki, J., & Jaskula, P. (2016). Analysis of effect of over-
loaded vehicles on fatigue life of flexible pavements based on
weigh in motion (WIM) data. _International Journal of Pavement
Engineering_ , _17_ (8), 716–726.https://doi.org/10.1080/10298436.2015.
1019493
Shah, R., Sharma, Y., Mathew, B., Kateshiya, V., & Parmar, J.
(2016). Review paper on overloading effect. _International Jour-
nal of Advanced Scientific Research and Management_ , _1_ (4), 131–
134.
Simonyan, K., & Zisserman, A. (2015). Very deep convolutional
networks for large-scale image recognition. In _3rd international
conference on learning representations (ICLR 2015)_ (pp. 1–14). San
Diego, CA, USA, May 7–9, 2015, Conference Track Proceedings.
[http://arxiv.org/abs/1409.](http://arxiv.org/abs/1409.)
Slootmans, F. (2023). _Facts and figures—Buses / coaches / heavy
goods vehicles—2023_. Technical report, European Commission,
Directorate General for Transport, Brussels, Belgium.
Sujon, M., & Dai, F. (2021). Application of weigh-in-motion technolo-
giesforpavementandbridgeresponsemonitoring:State-of-the-art
review. _Automation in Construction_ , _130_ , 103844.https://doi.org/
10.1016/j.autcon.2021.
Wang, Z., Zhu, J., & Ma, T. (2024). Deep learning–based detection
of vehicle axle type with images collected via UAV. _Journal of
Transportation Engineering, Part B: Pavements_ , _150_ (3), 04024032.
https://doi.org/10.1061/JPEODX.PVENG-
Xu, Z., Wei, B., & Zhang, J. (2023). Reproduction of spatial–temporal
distributionoftrafficloadsonfreewaybridgesviafusionofcamera
video and ETC data. _Structures_ , _53_ , 1476–1488.https://doi.org/10.
1016/j.istruc.2023.05.

```
Yan, W., Ren, H., Luo, X., & Li, S. (2025). Hybrid-data-driven
bridge weigh-in-motion technology using a two-level sequential
artificial neural network. Computer-Aided Civil and Infrastruc-
ture Engineering , 40 (20), 2992–3012.https://doi.org/10.1111/mice.
13415
Yang, J., Bao, Y., Sun, Z., & Meng, X. (2024). Computer vision-based
real-time identification of vehicle loads for structural health mon-
itoring of bridges. Sustainability , 16 (3), 1081.https://doi.org/10.
3390/su
Zhang, H., Zhu, J., Zhou, Y., & Shen, Z. (2024). Non-contact weigh-
in-motion approach with an improved multi-region of interest
method. Mechanical Systems and Signal Processing , 212 , 111323.
https://doi.org/10.1016/j.ymssp.2024.
Zhang, J., Kong, X., OBrien, E. J., Peng, J., & Deng, L. (2023).
Noncontact measurement of tire deformation based on com-
puter vision and Tire-Net semantic segmentation. Measure-
ment , 217 ,113034.https://doi.org/10.1016/j.measurement.2023.
113034
Zhang, J., OBrien, E. J., Kong, X., & Deng, L. (2024). Factors affecting
the accuracy of a computer vision-based vehicle weight measure-
ment system. Measurement , 224 ,113840.https://doi.org/10.1016/j.
measurement.2023.
Žnidarič, A., & Kalin, J. (2020). Using bridge weigh-in-motion sys-
tems to monitor single-span bridge influence lines. JournalofCivil
Structural Health Monitoring , 10 (5), 743–756.https://doi.org/10.
1007/s13349-020-00407-
Žnidarič, A., Kalin, J., & Kreslin, M. (2018). Improved accuracy
and robustness of bridge weigh-in-motion systems. Structure and
InfrastructureEngineering , 14 (4),412–424.https://doi.org/10.1080/
15732479.2017.
```
```
How to cite this article: Šoberl, D., Kalin, J.,
Anžlin, A., Kreslin, M., Čopič Pucihar, K., Kljun,
M., Hekič, D., & Žnidarič, A. (2025). Enhanced
precision in axle configuration inference for bridge
weigh-in-motion systems using computer vision
and deep learning. Computer-Aided Civil and
Infrastructure Engineering , 40 ,6201–6216.
https://doi.org/10.1111/mice.
```