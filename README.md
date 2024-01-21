\documentclass[12pt, journal]{IEEEtran}

%Headers
\usepackage[dvips]{graphicx}    %package that does pdfs
\usepackage{color}              %this needs to be here also
\usepackage{subcaption}
\usepackage{footnote}
\usepackage{cite}
\usepackage{url}
\usepackage{hyperref}
\usepackage{amsmath}
%\usepackage{tikz}
%\usetikzlibrary{positioning,fit,arrows.meta,backgrounds}
%\usepackage[utf8]{inputenc}
%%%%<
%\usepackage{verbatim}
%%%%>
%\usetikzlibrary{shapes.geometric,arrows}
%
%\usepackage[margin=0.1in]{geometry}
%\tikzset{
%    module/.style={%
%        draw, rounded corners,
%        minimum width=#1,
%        minimum height=7mm,
%        font=\sffamily
%        },
%    module/.default=2cm,
%    >=LaTeX
%}

\providecommand{\keywords}[1]
{
  \small	
  \textbf{\textit{Keywords---}} #1
}

\title{Image-Guided Object Detection using OWL-ViT and Enhanced Query Embedding Extraction}
\author{Melih Serin\textsuperscript{1}\thanks{\textsuperscript{1}Senior Student, Department of Electrical and Electronics Engineering, Boğaziçi University.  Email: melihsrnn@gmail.com}
}
\date{January, 2024}

\begin{document}
\maketitle

\begin{abstract}
\textit{Computer vision has been receiving increasing attention with the recent complex Generative AI models released by tech industry giants, such as OpenAI and Google. However, there is a specific subfield that we wanted to focus on, that is, Image-Guided Object Detection. A detailed literature survey directed us towards a successful study called Simple Open-Vocabulary Object Detection with Vision Transformers (OWL-ViT)\cite{OWL_ViT}, which is a multifunctional complex model that can also perform image-guided object detection as a side function. In this study, some experiments have been conducted utilizing OWL-ViT architecture as the base model and manipulated the necessary parts to achieve a better one-shot performance.}
\end{abstract}

\keywords{Computer Vision, OWL-ViT, Object Detection, Vision Transformers, End-to-End Training, gIoU Loss, L1 Loss, CLIP, DETR,Cosine Similarity, Yolov8, MS-COCO}

\section{Introduction}
Recently, there have been numerous developments in the use of artificial neural networks, which have expanded the areas of study for Artificial Intelligence. Especially in computer vision, a rapid increase has occurred in a number of distinct real-life applications since researchers discovered a huge range of new utilization areas for complex computer vision models.

These applications include segmentation and object detection. As they constantly work to improve the functionality of these models, some of them now accept multimodal queries such as markers and/or points, masks, and texts. For instance, a new model called Segment Anything (SAM)\cite{SAM} was developed by Meta-AI to undergo this kind of multi-functionality. These improvements had a significant positive effect on the capability of zero-shot learning in computer vision models. Zero-shot learning is, as the name states, the model’s ability to predict unseen data samples. In other words, it shows how generalizable the model is to its problem definition. This term has been used in many studies.

%\begin{figure}[h]
%\begin{center}
%\includegraphics[width=\linewidth, height=4cm]{C:/Users/User/Desktop/Senior Project/Latex_Report/SimpleDocument/DETR.png}
%\caption{Architecture of DETR}
%\label{fig:DETR}
%\end{center}
%\end{figure}

After a detailed literature survey, some studies similar to this one are found. One of them is Simple Open-Vocabulary Object Detection with Vision Transformers (OWL-ViT)\cite{OWL_ViT}. In OWL-ViT, they designed an open-vocabulary object detection method based only on transformer encoders (without any decoders). The model’s pre-training process is similar to that of CLIP\cite{CLIP} training, which is essentially contrastive image-text pre-training on image-text representations. They then performed end-to-end training that was almost equivalent to DETR\cite{DETR}. The model is agnostic to the source of the query representations because the image and text components of the model are not fused. Utilizing this aspect, they also used the model for one-shot image-conditioned object detection, which detects certain objects in an image using image query embeddings.

In this study, the aim is to train a model that can detect specific objects in an image defined by image queries that are not of categories fed into the model during training. A model is desired such that it is as generalizable as possible, which means that the model should detect objects not previously seen. To achieve this, several experiments are conducted using the OWL-ViT as the base model.  

\begin{figure*}[h]
\begin{center}
\includegraphics[width=\linewidth, height=10cm]{C:/Users/User/Desktop/Senior Project/Latex_Report/SimpleDocument/OWL_ViT_Architecture.png}
\caption{Architecture of OWL-ViT for Image-Guided Object Detection}
\label{fig:OWL_ViT_Architecture}
\end{center}
\end{figure*}

\section{Background}

\subsection{DETR\cite{DETR}}
Detection Transformers (DETR) presented a new method that considers object detection as a direct set prediction problem. The approach streamlines the detection pipeline, effectively removing the need for many hand-designed components, such as a non-maximum suppression procedure or anchor generation, that explicitly encode our prior knowledge about the task. The main components of the new framework are a set-based global loss that forces unique predictions via bipartite matching and a transformer encoder-decoder architecture. Given a fixed small set of learned object queries, DETR explains the relations between the objects and the global image context to directly output the final set of predictions in parallel. 
This model is conceptually simple and does not require a specialized library, unlike many other modern detectors. DETR demonstrates accuracy and run-time performance on par with the well-established and highly optimized Faster R-CNN\cite{faster_rcnn} baseline on the challenging COCO\cite{3} object detection dataset. 

%\begin{figure}[h]
%\begin{center}
%\includegraphics[width=\linewidth, height=4cm]{C:/Users/User/Desktop/Senior Project/Latex_Report/SimpleDocument/CLIP.png}
%\caption{Summary of CLIP Approach}
%\label{fig:CLIP}
%\end{center}
%\end{figure}


\subsection{CLIP\cite{CLIP}}
CLIP (Contrastive Language-Image Pre-training) is pre-trained to predict if an image and a text snippet are paired together in its dataset. For each dataset, the names of all classes in the dataset were used as the set of potential text pairings and predicted the most probable (image, text) pair.

The image encoder inside the model is a computer vision backbone that computes a feature representation for the image, and the text encoder is a hypernetwork that generates the weights of a linear classifier based on the text specifying the visual concepts represented by the classes.


\begin{figure}[h]
\begin{center}
\includegraphics[width=\linewidth, height=4cm]{C:/Users/User/Desktop/Senior Project/Latex_Report/SimpleDocument/vit_architecture.jpg}
\caption{Architecture of ViT}
\label{fig:ViT}
\end{center}
\end{figure}

\subsection{Vision Transformers\cite{ViT}}

While the Transformer\cite{Transformer} architecture has become the de facto standard for natural language processing tasks, its applications to computer vision remain limited. In vision, attention is either applied in conjunction with convolutional networks, or used to replace certain components of convolutional networks while maintaining their overall structure in place. We show that this reliance on CNNs\cite{CNN} is not necessary, and that a pure transformer applied directly to sequences of image patches can perform very well on image classification tasks as in Figure~\ref{fig:ViT}\cite{1}. When pre-trained on large amounts of data and transferred to multiple mid-sized or small image recognition benchmarks (ImageNet\cite{ImageNet}, CIFAR-100\cite{CIFAR}, VTAB\cite{VTAB}, etc.), the ViT attains excellent results compared to state-of-the-art convolutional networks and requires substantially fewer computational resources to train. 

\begin{figure*}[h]
\begin{center}
\includegraphics[width=\linewidth, height=5cm]{C:/Users/User/Desktop/Senior Project/Latex_Report/SimpleDocument/CNN_network.png}
\caption{Proposed Query Embedding Extraction}
\label{fig:query_embed}
\end{center}
\end{figure*}



%Illustrate inserting a table
\begin{table*}
	\begin{center}
	\begin{tabular}{p{3cm} p{2cm} p{2cm} p{2cm} p{2cm}}
	\hline
	\textbf{Name} & \textbf{Train} & \textbf{Seen Test} & \textbf{Unseen Test} & \textbf{Categories} \\
	\hline
	Digit Detection & 280 & 120 & 30 & 10 \\

	Object Detection & 400 & 100 & 25 & 5 \\
	\hline
	\end{tabular}
	\caption{Customized Datasets used in this study.}
	\label{table:Datasets}
	\end{center}
\end{table*}

\section{Method}

The objective is to study visual query encoding for deep-object detection models. Nowadays, computer vision models accept multimodal queries, such as images, text, and/or markers, to perform tasks, such as segmentation, as in the case of SAM. In this study, we aim to explore the possibility of training an object detection model with visual queries to train not only the object search/detect module but also the semantic query encoding layers. OWL-ViT is used as the base model to train a one-shot image-guided object detection model on the customized datasets.

\subsection{Model Architecture}
In the architecture of OWL-ViT, two fundamental parts undergo the same procedure. These parts extract query and image embeddings from the query and target images successively.

To extract query embeddings, a vision transformer is used to obtain the feature vectors (tokens). The tokens contain the information for each image patch. These tokens are then fed into three different processes. First, by linearly projecting tokens onto the embedding dimension, class-embedding vectors for each patch are obtained. Consecutively, tokens go through two different Multilayer Perceptrons (MLPs)\cite{MLP}. While one calculates a score showing the probability that each token is an object (objectness score), the other finds the corresponding bounding box coordinates for each token. A bias is added to the state location of each patch for the second MLP to return meaningful location coordinates. Finally, the top k class embeddings were chosen based on their objectness scores and used as query embeddings for the target image. 

In the other part, the target image passes through the same pre-trained vision transformer. After the composition of image feature vectors, tokens are fed only into the linear projection layer and coordinate calculating MLP because we want to select the necessary image embeddings based on their similarity to query embeddings, not on objectness scores. To match the image and query embeddings, Cosine Similarity is used to obtain output logit scores for each image embedding. Subsequently, the output object(s) are defined as those with the highest scores.

Although the architecture of OWL-ViT is conceptually simple, it may lead to an overkill if the objective is object detection with image queries because the image queries are allowed to be small in size and chosen as basic as possible in order them to visually represent the corresponding object in a better way. Therefore, in one of the experiments, the part where the extraction of query embeddings occurs, is simplified to output only one representative embedding from the object query. This experiment uses a CNN network that is analogous to the one in \cite{10} and has been pre-trained to get a representative embedding for each object query image. Further information about this network is provided in the Model Details section.


\subsection{Training}

The OWL-ViT method involves a contrastive pre-training of images and texts, which is accomplished using Vision Transformers (ViT) with Multihead Attention Pooling (MAP)\cite{4}. In this work, we made use of one of the pre-trained OWL-ViT models. 

To ensure good performance, a series of training experiments are conducted with parameters close to those reported in the OWL-ViT paper. As the focus of this study is not on the object classification component, the loss used is tailored so that only the object detection function is optimized, for which the bipartite matching loss of DETR was employed with the necessary task-specific normalizations.

\section{Experiments}

\subsection{Model Details}

ViT's working process begins with a given image being divided into many patches, the size of which is specified by the ViT-B/32 model. To suit this model, the input image must first be preprocessed. Assuming we have a 650x900 image, it must be resized to 768x768 pixels. The encoder then divides this image into 576 patches of 32x32, with each patch's feature representation vector (token) being of size 1x1x1024.

%Illustrate inserting a table
\begin{table*}
	\begin{center}
	\begin{tabular}{p{5cm} p{2cm} p{2cm}}
	\textbf{Object Detection} \\
	\hline
	\text{Method} & \text{Seen Test} & \text{Unseen Test} \\
	\hline
	OWL-ViT(ViT-B/32) & 73.5 & 58.3 \\
	OWL-ViT(ViT-B/32)(ours) & \textbf{86.2} & \textbf{80.0} \\
	OWL-ViT+CNN(ours) & 79.5 & 68.2 \\
	\hline
	\textbf{Digit Detection} \\
	\hline
	OWL-ViT(ViT-B/32) & 10.8 & 19.2 \\
	OWL-ViT(ViT-B/32)(ours) & \textbf{19.6} & 15.0 \\
	OWL-ViT+CNN(ours) & 13.5 & \textbf{20.0} \\
	\hline
	\end{tabular}
	\caption{AP50 Performance of Image-Conditioned Detection Models}
	\label{table:Results}
	\end{center}
\end{table*}


The query image is run through three processes - a linear projection layer to create embeddings of size 1x1x512, a MLP for objectness score and another MLP to return the bounding box coordinates for each token. For the bounding box MLP, center location information of each patch is fed into the MLP as biases to get more accurate bounding boxes. In this case, as the query image contains one object only, the objectness score MLP is omitted and the embedding with the biggest bounding box is chosen as the query embedding. 

The target image is also put through the linear projection layer and the MLP for box coordinates. Finally, the Cosine Similarity as in the Equation~\ref{eqn:Cos}\cite{5} is used to compare the target image embeddings to the query embedding and the highest scoring bounding box is the output object in the target image.

\begin{center}
\begin{equation}
	\cos(\theta) = \frac{\sum_{i}^{n} A_{i}B_{i}}{\sqrt{\sum_{i}^{n} A_{i}^{2}}\sqrt{\sum_{i}^{n} B_{i}^{2}}}
	\label{eqn:Cos}
\end{equation}
\end{center}


Moreover, in order to prevent an overabundance of query embedding extraction, it was opted to utilize a pre-trained CNN Network architecture which only works on one patch at a time instead of 576 different patches just like in Figure~\ref{fig:query_embed}. One embedding that is created to describe an entire digit or another kind of object query image is generated by this process. The embedding is pre-trained on digit and object detection query images independently for approximately 50 epochs.


\subsection{Dataset}

In order to carry out this study, we generated two datasets tailored for this purpose. It should be noted that the pre-trained model applied as the base model was pre-trained on the LiT\cite{11}, O365\cite{12}, and VG\cite{13} datasets.

The first dataset was intended for digit detection and consisted of 10 labeled door number images and five digit images for each digit class, with classes 1 and 9 only used in the test set. The training set was composed of 280 pairs of target images and objects, with the test set on seen classes being 120 pairs and test set on unseen classes being 30 pairs as shown in the Table~\ref{table:Datasets}. 

The other dataset contained some of the MS-COCO classes, such as Horse, Car, Person, Airplane, and Elephant, with Elephant chosen as the unseen object class. There were five labeled target images and objects for each class, making the training set 400 pairs and unseen and seen test sets 25 and 100 pairs respectively as shown in the Table~\ref{table:Datasets}. 

Additionally, target images in the customized MS-COCO dataset were labeled using Yolov8\cite{7}, a model that is successful for object detection.

\subsection{Hyperparameters}

Hyperparameter settings used in this study are as following
\begin{itemize}
  \item Learning rate = $10^{-6}$
  \item Weight Decay = $10^{-4}$
  \item Equal weights for bounding box and gIoU losses
\end{itemize}

\subsection{Loss}

The bipartite matching loss introduced by the DETR is used, which is composed of L1 loss and gIoU loss. The L1 loss is utilized to reduce the discrepancy between the target and the predicted bounding box by summing up all the absolute differences as in the Equation~\ref{eqn:L1}\cite{8}. To compute the Generalized Intersection over Union, two boxes A and B are firstly enclosed in the smallest convex shape C in S in $R^{n}$. Afterwards, the ratio of C excluding A and B is divided by the area of C as in the Equation~\ref{eqn:gIoU}\cite{9}.

\begin{center}
\begin{equation}
	L1 = \sum_{i}^{n}|y_{true} - y_{predicted}|
	\label{eqn:L1}
\end{equation}
\end{center}

\begin{center}
\begin{equation}
	gIoU = \frac{\lvert A \cap B\rvert}{\lvert A \cup B\rvert } - \frac{\lvert C\backslash(A \cup B)\rvert}{\lvert C \rvert}
	\label{eqn:gIoU}
\end{equation}
\end{center}


\subsection{Model Performances}
As described above, the purpose is to identify corresponding objects in a target image to a query image. Table~\ref{table:Datasets} shows the distribution of our custom datasets that contain target and query images as pairs. Evaluation is done with the AP50 metric as they did it in \cite{OWL_ViT} and \cite{ting}. To compare our results, the results of the pre-trained model we used are also presented in Table~\ref{table:Results}. The train and test losses are shown in Figure~\ref{fig:losses}.
Both Digit and Object Detection results clearly demonstrate the improvement of the performance of our image-guided object detection model on both seen and unseen test sets. Additionally, as seen in Table~\ref{table:Results}, the OWL-ViT+CNN model has an edge when tested with unseen categories after it has been trained on the Digit Detection dataset. In contrast, the original OWL-ViT model is more adept at working with the COCO categories in the Object Detection dataset.

\section{Realistic Constraints}
The primary obstacle we face is technical in nature. Developing intricate models necessitates GPUs and improved computers, which we don't have access to. We have been utilizing Google Colab during our work, though it is not the pro version so it has its own restrictions. We did our best to tackle this difficulty.

\subsection{Social, Environmental and Economic Impact}
We hope that this study is a positive step towards achieving one-shot image-guided object detection.

\subsection{Cost Analysis}
For the project, the duration and probable expenses for an online processor were essential.
\subsection{Standards}
The project will be held with special emphasis to IEEE, IET, EU and Turkish standards. Engineering code of conduct will be followed.
\section{Conclusion}
In summary, we employed OWL-ViT as the basis to construct a model of image-aided object detection which is also capable of detecting objects it has not encountered yet. Several experiments were conducted in which certain modifications were made to the original OWL-ViT model design to boost its performance. Based on the results, we are confident that we have taken a significant step forward.


\section*{Acknowledgements}
Over the course of the study, I had a great experience working closely with Burak ACAR and Zafer DOĞAN. I owe them my gratitude for guiding me and providing me with assistance during this process. I would like to express my appreciation for their help.

%Illustrate creating a bibliography by referring to an external bibliography file (SampleBibliography.bib).  Note that you must chose a style to determine how the bibliography is presented
\bibliography{SampleBibliography}
\bibliographystyle{aiaa}


\newpage
\onecolumn
%\section*{Appendix}
\appendix
\subsection{Losses}
\begin{figure}[h]
\captionsetup[subfigure]{labelformat=empty}
\begin{subfigure}{0.5\textwidth}
\includegraphics[width=\linewidth, height=7.5cm]{C:/Users/User/Desktop/Senior Project/Latex_Report/SimpleDocument/regular_od.png}
\caption{OWL-ViT on Object Detection Dataset}
\label{fig:digit6}
\end{subfigure}
\begin{subfigure}{0.5\textwidth}
\includegraphics[width=\linewidth, height=7.5cm]{C:/Users/User/Desktop/Senior Project/Latex_Report/SimpleDocument/modified_od.png}
\caption{OWL-ViT+CNN on Object Detection Dataset}
\label{fig:digit7}
\end{subfigure}
\begin{subfigure}{0.5\textwidth}
\includegraphics[width=\linewidth, height=7.5cm]{C:/Users/User/Desktop/Senior Project/Latex_Report/SimpleDocument/regular_dd.png}
\caption{OWL-ViT on Digit Detection Dataset}
\label{fig:digit8}
\end{subfigure}
\begin{subfigure}{0.5\textwidth}
\includegraphics[width=\linewidth, height=7.5cm]{C:/Users/User/Desktop/Senior Project/Latex_Report/SimpleDocument/modified_dd.png}
\caption{OWL-ViT+CNN on Digit Detection Dataset}
\label{fig:lossess}
\end{subfigure}

\caption{Losses}
\label{fig:losses}
\end{figure}


\newpage
\subsection{Qualitative Examples}
The figures below shows query object images on the right while target images are situated on the left.


\begin{figure}[h]
\captionsetup[subfigure]{labelformat=empty}
\begin{subfigure}{0.5\textwidth}
\includegraphics[width=0.9\linewidth, height=4cm]{C:/Users/User/Desktop/Senior Project/Latex_Report/SimpleDocument/regular_od_seen_image1.png}
\caption{Person}
\label{fig:digit6}
\end{subfigure}
\begin{subfigure}{0.5\textwidth}
\includegraphics[width=0.9\linewidth, height=4cm]{C:/Users/User/Desktop/Senior Project/Latex_Report/SimpleDocument/regular_od_seen_image2.png}
\caption{Horse}
\label{fig:digit7}
\end{subfigure}
\begin{subfigure}{0.5\textwidth}
\includegraphics[width=0.9\linewidth, height=4cm]{C:/Users/User/Desktop/Senior Project/Latex_Report/SimpleDocument/regular_od_seen_image3.png}
\caption{Airplane}
\label{fig:digit8}
\end{subfigure}
\begin{subfigure}{0.5\textwidth}
\includegraphics[width=0.9\linewidth, height=4cm]{C:/Users/User/Desktop/Senior Project/Latex_Report/SimpleDocument/regular_od_seen_image4.png}
\caption{Car}
\label{fig:digit9}
\end{subfigure}

\caption{Previously Seen Query Examples for Object Detection}
\label{fig:regular_od_1}
\end{figure}

\begin{figure}[h]
\captionsetup[subfigure]{labelformat=empty}
\begin{center}
\begin{subfigure}{0.5\textwidth}
\includegraphics[width=0.9\linewidth, height=4cm]{C:/Users/User/Desktop/Senior Project/Latex_Report/SimpleDocument/regular_od_unseen_image3.png}
\caption{Elephant}
\label{fig:digit9}
\end{subfigure}
\end{center}
\caption{Unseen Query Example for Object Detection}
\label{fig:regular_od_2}
\end{figure}



\begin{figure}[h]
\captionsetup[subfigure]{labelformat=empty}
\begin{subfigure}{0.5\textwidth}
\includegraphics[width=0.9\linewidth, height=4cm]{C:/Users/User/Desktop/Senior Project/Latex_Report/SimpleDocument/regular_dd_seen_image1.png}
\caption{Person}
\label{fig:digit6}
\end{subfigure}
\begin{subfigure}{0.5\textwidth}
\includegraphics[width=0.9\linewidth, height=4cm]{C:/Users/User/Desktop/Senior Project/Latex_Report/SimpleDocument/regular_dd_seen_image2.png}
\caption{Horse}
\label{fig:digit7}
\end{subfigure}
\begin{subfigure}{0.5\textwidth}
\includegraphics[width=0.9\linewidth, height=4cm]{C:/Users/User/Desktop/Senior Project/Latex_Report/SimpleDocument/regular_dd_seen_image3.png}
\caption{Airplane}
\label{fig:digit8}
\end{subfigure}
\begin{subfigure}{0.5\textwidth}
\includegraphics[width=0.9\linewidth, height=4cm]{C:/Users/User/Desktop/Senior Project/Latex_Report/SimpleDocument/regular_dd_seen_image4.png}
\caption{Car}
\label{fig:digit9}
\end{subfigure}
\begin{subfigure}{0.5\textwidth}
\includegraphics[width=0.9\linewidth, height=4cm]{C:/Users/User/Desktop/Senior Project/Latex_Report/SimpleDocument/regular_dd_seen_image5.png}
\caption{Car}
\label{fig:digit9}
\end{subfigure}
\begin{subfigure}{0.5\textwidth}
\includegraphics[width=0.9\linewidth, height=4cm]{C:/Users/User/Desktop/Senior Project/Latex_Report/SimpleDocument/regular_dd_seen_image6.png}
\caption{Car}
\label{fig:digit9}
\end{subfigure}

\caption{Previously Seen Query Examples for Digit Detection}
\label{fig:regular_dd_1}
\end{figure}


\begin{figure}[h]
\captionsetup[subfigure]{labelformat=empty}
\begin{center}
\begin{subfigure}{0.5\textwidth}
\includegraphics[width=0.9\linewidth, height=4cm]{C:/Users/User/Desktop/Senior Project/Latex_Report/SimpleDocument/regular_dd_unseen_image1.png}
\caption{Elephant}
\label{fig:digit9}
\end{subfigure}
\end{center}
\caption{Unseen Query Example for Digit Detection}
\label{fig:regular_dd_2}
\end{figure}




\clearpage
\newpage
\subsection{Failures}
\begin{figure}[h]
\captionsetup[subfigure]{labelformat=empty}
\begin{subfigure}{0.5\textwidth}
\includegraphics[width=0.9\linewidth, height=4cm]{C:/Users/User/Desktop/Senior Project/Latex_Report/SimpleDocument/dd_failures_seen1.jpg}
\caption{Eight}
\label{fig:digit6}
\end{subfigure}
\begin{subfigure}{0.5\textwidth}
\includegraphics[width=0.9\linewidth, height=4cm]{C:/Users/User/Desktop/Senior Project/Latex_Report/SimpleDocument/dd_failures_seen2.jpg}
\caption{Seven}
\label{fig:digit7}
\end{subfigure}
\begin{subfigure}{0.5\textwidth}
\includegraphics[width=0.9\linewidth, height=4cm]{C:/Users/User/Desktop/Senior Project/Latex_Report/SimpleDocument/dd_failures_seen3.jpg}
\caption{Six}
\label{fig:digit8}
\end{subfigure}
\begin{subfigure}{0.5\textwidth}
\includegraphics[width=0.9\linewidth, height=4cm]{C:/Users/User/Desktop/Senior Project/Latex_Report/SimpleDocument/dd_failures_seen4.jpg}
\caption{Zero}
\label{fig:digit9}
\end{subfigure}
\begin{subfigure}{0.5\textwidth}
\includegraphics[width=0.9\linewidth, height=4cm]{C:/Users/User/Desktop/Senior Project/Latex_Report/SimpleDocument/dd_failures_seen5.jpg}
\caption{Six}
\label{fig:digit9}
\end{subfigure}
\begin{subfigure}{0.5\textwidth}
\includegraphics[width=0.9\linewidth, height=4cm]{C:/Users/User/Desktop/Senior Project/Latex_Report/SimpleDocument/od_failures_seen.jpg}
\caption{Airplane}
\label{fig:digit9}
\end{subfigure}

\caption{Previously Seen Query Failures}
\label{fig:failures_1}
\end{figure}


\begin{figure}[h]
\captionsetup[subfigure]{labelformat=empty}

\begin{subfigure}{0.5\textwidth}
\includegraphics[width=0.9\linewidth, height=4cm]{C:/Users/User/Desktop/Senior Project/Latex_Report/SimpleDocument/dd_failures_unseen.jpg}
\caption{Nine}
\label{fig:digit9}
\end{subfigure}
\begin{subfigure}{0.5\textwidth}
\includegraphics[width=0.9\linewidth, height=4cm]{C:/Users/User/Desktop/Senior Project/Latex_Report/SimpleDocument/od_failures_unseen.jpg}
\caption{Elephant}
\label{fig:digit9}
\end{subfigure}


\caption{Unseen Query Failures}
\label{fig:failures_2}
\end{figure}



\end{document}

