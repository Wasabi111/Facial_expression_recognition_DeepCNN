# Expression Recognition using SVM with OpenFace

The theory of using Face Action Coding system to predict human facial expressions could be found:
https://en.wikipedia.org/wiki/Facial_Action_Coding_System#cite_note-pmid18020726-8
http://www.cs.cmu.edu/~jeffcohn/pubs/Reed%20et%20al.%202007.pdf

## OpenFace 
https://github.com/TadasBaltrusaitis/OpenFace

OpenFace software provides action units detection and it could be used in real-time senarios. 

It also provides action unit informations.

![](OpenFaceSVM/135.png)
## SVM
https://zhuanlan.zhihu.com/p/31886934
Here provides 4 different kernal: Linear, Linear SVC, RBF and polynomial.
Training and Testing data are shuffled.

## Dependencies

* sklearn
* OpenFace
* OpenCV

## Example 
0. Using OpenFace to produce action units detection data for pictures or videos.

1. Prepare AU data and mood label for SVM learning (you may use CK+ dataset) 
```
python sklearnSVMSample
```
Here is a classified version of a part of CK+ dataset: https://pan.baidu.com/s/1bBdz2KkGz-3Dg6-PA-3HIw extract code: o97n

You could see classifed results:

![](OpenFaceSVM/Figure_1-1.png)




