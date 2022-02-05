# Group Homework 1

Multi-Layer Perceptron.

## Description

ให้เวลา 2 weeks สำหรับเก็บคะแนน 10% (ส่งภายในวันศุกร์ ก่อนเริ่ม class05)
โดยให้หา tabular data (structured data) ที่น่าสนใจ มาลองทำ MLP เพื่อเทียบกับ Traditional ML (อย่างน้อย 1 อัน) ว่าแตกต่างกันอย่างไรในแง่ต่าง ๆ และบอกเพิ่มเติมว่า ทั้ง 2 อัน มีข้อดี ข้อเสีย อย่างไร รวมถึงบอกรายละเอียดและ % contribute ของสมาชิปแต่ละคน
ส่งผ่าน : Github (1 กลุ่มต่อ 1 url)

# Conclusion

## Traditional Machine Learning
1. จาก EDA เราพบว่าดาต้า imbalance แต่ละ feature ไม่ค่อยมีความสัมพันธ์กัน
2. ลองทำ classification 3 models (SVM,RF,XGB) RF ได้ผล test acc สูงสุด (~0.69)
3. คิดว่าเนื่องจาก imbalance data ทำให้ class ที่มี data น้อยๆ (class 3,4,8) มีจำนวนdataมาให้เทรนน้อยเกินไปเลยทำให้ทายไม่ค่อยถูก
4. ลอง oversampling เพื่อ balance data แต่ก็ไม่ช่วยให้ accurency ดีขึ้นได้

## Multi-Layer Perceptron

### Turning parameter:

* เรา keep จำนวน epoch = 20 เนื่องจากทางกลุ่มได้ดูแล้วว่า แค่ 20 epoch ก็จะเห็นว่าแนวโน้มของ accuracy value และ loss value นั้นลู่เข้าสู่ค่าๆหนึ่ง ซึ่งกราฟเริ่ม stable แล้ว
* ทางกลุ่ม keep batch size = 100 เนื่องจาก หลังจากที่ทางกลุ่มของ vary ค่าของ batch size แล้ว พบว่าไม่มีผลกับ accuracy และ loss ของโมเดล แต่จะไปมีผลกับ run time แต่เนื่องจากข้อมูลของเราไม่ได้ใหญ่มาก ผลต่างของเวลาที่ใช้ในการ run จึงไม่ค่อยเห็นผลอย่างมีนัยยะสำคัญ
* Optimizer ทางกลุ่มไม่ได้ปรับเปลี่ยนอะไร เนื่องจากทางกลุ่มคิดว่า optimizer มีผลต่อความเร็ว และลักษณะในการลู่เข้าสู่ optimal point หรือบาง optimizer อาจจะช่วยแก้ปัญหาที่ไปติดอยู่ใน flat region หรือ การลู่เข้าที่ช้า แต่ไม่มีผลต่อ accuracy

### Training data set methodology

1.	Validation split rate :arrow_right: ทางกลุ่มได้ลองปรับตั้งแต่ 0.1 จนถึง 0.9 พบว่า ไม่มีผลกับ performance ของ model
2.	Dropout ratio :arrow_right: ทางกลุ่มได้ทำการทดลองตั้งแต่ dropout ratio = 0.0 จนถึง 0.9 พบว่า ที่ dropout ratio ต่ำๆ (0.0 -0.3) จะให้ performance ที่ดี และ performance จะลดลง และยิ่ง dropout ratio เข้าใกล้ 0.9 ก็จะยิ่งให้ performance ที่ต่ำลง
3.	ปรับจำนวนของ hidden layer :arrow_right: เนื่องจากทางกลุ่มเห็นว่าจำนวนของ hidden layer พี่เพิ่มมากขึ้นน่าจะทำให้ accuracy ดีขึ้น โดยเราได้ลองปรับจำนวนของ hidden layer ตั้งแต่ 1-9 แล้วพบว่าที่จำนวน hidden layer เท่ากับ 1 และ 2 นั้นให้ accuracy ของ training data ประมาณ 60% และ 80% ตามลำดับ แต่เมื่อลองเพิ่มจำนวน hidden layer ขึ้นจะพบว่า จำนวน hidden layer ตั้งแต่ 3-9 นั้น จะให้ accuracy ของ training data อยู่ที่ประมาณ 90-95% แต่อย่างไรก็ตาม ค่าเริ่ม stable อยู่ที่ประมาณ 93% หรือก็คือ จำนวนของ hidden layer ที่เพิ่มขึ้นมากกว่านี้ก็ไม่สามารถทำให้ accuracy ของ training data ดีขึ้นไปกว่านี้ได้อีกแล้ว
ในส่วนของ loss ก็เช่นกัน ยิ่งจำนวน hidden layer เพิ่มมากขึ้น ค่า loss ก็จะลดต่ำลง โดยค่า loss ที่ต่ำที่สุดของ training data จะอยู่ที่ 0.1204 ในขณะที่ loss ของ test data จะอยู่ที่ประมาณ 1.9
4.	ปรับจำนวน node ในแต่ละ hidden layer (เรากำหนดให้ในแต่ละ hidden layer มีจำนวน node เท่ากัน)  โดยทางกลุ่มของเราได้ลองเริ่มปรับจำนวน node ตั้งแต่ 10, 20, 50, 100, 200, 300 และ 1000 ในช่วงแรกที่จำนวน node น้อยๆ การเพิ่มจำนวน node จะทำให้ performance ของ model ดีขึ้นมาก (accuracy เพิ่มขึ้น, loss ลดลง) แต่อย่างไรก็ตาม เมื่อเพิ่มจำนวน node ไปถึงจุดหนึ่ง performance ก็ไม่ได้ดีขึ้นอีกแล้ว ในกรณีนี้เราได้ทดลอง จะพบว่าที่จำนวน node 300 และ 1000 ไม่ทำให้ accuracy ต่างกัน
5.	Activation function :arrow_right: ทางกลุ่มได้ลองใช้ activation function แบบต่างๆ (ตัวอย่างเช่น Relu, Sigmoid, Tanh, etc.) พบว่าที่ condition เดียวกัน Relu จะให้ performance ที่ดีที่สุด (ทั้งในส่วนของ training data และ test data)
6.	Regularization :arrow_right: จากที่ทางกลุ่มได้ลองปรับ turning parameter ข้างต้น (จำนวน node, จำนวน hidden layer, activation function) พบว่า performance ที่ดีขึ้นนั้น จะเป็นในส่วนของ training data แต่ในขณะที่ accuracy ของ test data จะอยู่ที่ประมาณ 40-60% (loss อยู่ที่ 1-2 โดยประมาณ)ไม่ว่าจะปรับ turning parameter อย่างไร ก็ไม่สามารถ ทำให้ performance ดีขึ้นได้ ซึ่งทางกลุ่มคิดว่าเกิดจาก overfitting จึงได้ลองใส่ regularization เข้ามาเพื่อจะทำให้ test data มี accuracy ที่ดีขึ้นเข้าใกล้ training data มากขึ้น
แต่หลังจากที่ทางกลุ่มได้ลองใส่ parameter L1, L2 รวมทั้ง L1+L2 เข้าไป พบว่า ไม่ได้ช่วยดึงให้ accuracy ของ test data สูงขึ้น รวมถึงบาง ค่า α ของ regularization นั้น กลับดึงให้ค่า accuracy ของ training data ลดลงอีกต่างหาก

# Summary of ML vs MLP
Performance accurency ของ F1 Score ได้ผลที่ไกล้เคียงกันมาก จากการทดลองอาจจะยังไม่ได้ประโยชน์จากการใช้ Multi-Layer Perceptron
มากนักเพราะขนาดของข้อมูลและปัญหาไม่ได้ใหญ่และยากมากนัก จึงสามารถใช้เพียงแค่ Machine Learning ก็พอสามารถใช้ทำนายข้อมูลชุดที่เลือกมาได้

![image](https://github.com/khwanck/DeepLearning_NIDA01/blob/main/MLvsMLP.PNG)

## Team Members
ID   | Name | Responsibility |% Contribute
--------- | ------ | ------ | ------
6310422062 | Nattipon Itponjaroen | Tuning ML  | 20%
6310422066 | Kumpee Apsornpasakorn  | Tuning MLP | 20%
6310422067 | Eakarat Pimthai  | Tuning ML | 20%
6310422068 | Khwanchai Kaewkaisorn  | Tuning MLP | 20%
6310422070 | Shularp Panitchart | Tuning MLP | 20%
