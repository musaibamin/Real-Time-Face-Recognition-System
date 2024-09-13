CREATE DATABASE FaceRecognitionMusaib;

USE FaceRecognitionMusaib;


CREATE TABLE regteach(
fname VARCHAR(50),
lname VARCHAR(50),
cnum VARCHAR(20),
email VARCHAR(100) PRIMARY KEY,
ss_que VARCHAR(100),
ss_ans VARCHAR(100),
pwd VARCHAR(100)
);


CREATE TABLE student(
`Student_ID` int(10) Primary Key,
`Name` VARCHAR(20),
`Department` VARCHAR(50),
`Course` VARCHAR(50),
`Year` VARCHAR(7),
`Gender` VARCHAR(20),
`DOB` VARCHAR(20),
`Mobile_No` VARCHAR(100),
`Address` VARCHAR(500),
`Roll_No` VARCHAR(10),
`Email` VARCHAR(200),
`PhotoSample` VARCHAR(300)
);


CREATE TABLE attendancelog(
`entry_number` int(100) primary key auto_increment,
`Student_ID` int(10),
`Roll_No` VARCHAR(10),
`datestamp` VARCHAR(20),
`ts` VARCHAR(20),
foreign key(`Student_ID`) references STUDENT(`Student_ID`)
);