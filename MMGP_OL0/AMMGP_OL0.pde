 //<>//
RotatingFoil test; //<>//
SaveData dat;

float maxT;
String datapath;//输出数据保存的路径

int Re = 500, nflaps = 40;//拍动次数
float hc = 1, chord = 1.0;
//changed parameters
// theta0 5:5:40, AD = 0.1:0.1:0.6, Phi = 0:30:180, St = 0.1:0.04:03
float theta0 = 5, AD = 0.1, Phi=0, St = 0.05;//初始化运动参数
int Num = 1;
String Sp;
float CT = 0, CL = 0, CP = 0,Eta=0;
float y = 0, angle = 0;
int resolution = 16, xLengths=16, yLengths=16, zoom = 2;//zoom:画幅放大倍数
int picNum = 10;//图片数量
float tCurr = 0, tStep = .005;
int count=0;//OL新加入的变量
int testcount=1;
float nacaName1 = 0,nacaName2 = 0,nacaName3 = 15;//初始翼型

void settings(){
  size(zoom*xLengths*resolution, zoom*yLengths*resolution);//设置画幅尺寸
}
void setup() {
  datapath = "Re500"+"/" + str(nacaName1)+str(nacaName2) +str(nacaName3)+ '_' + "St="+str(St)+"_AD="+str(AD)+"_theta="+str(theta0)+"_Phi="+str(Phi);//文件保存路径

  setUpNewSim();
}
void draw() {
  
  test.update2();
  
  double angle=(test.foil.phi-PI)*180/PI;//单位为度数
  PVector forces = test.foil.pressForce(test.flow.p);
  PVector coordinate = test.foil.xc;
  y = coordinate.y - yLengths * resolution / 2.;
  PVector deltaxy = test.foil.dxc;
  float phivel = test.foil.dphi*resolution/test.dt; // with unit of rad/s
  float vely = deltaxy.y*resolution/test.dt;
  float M = test.foil.pressMomentpiv(test.flow.p, 1./4*resolution, 0);
    
  if(test.t>=maxT/20*0){
    count+=1;
    CT += -forces.x / resolution * 2;
    CP += ((-M*phivel/resolution)+(-forces.y*vely))*2/resolution;
    // CP += ((forces.y*vely))*2/resolution;
    CL += forces.y / resolution * 2;
    Eta=-CT/CP;
    print("CP"+(forces.y*vely)+"forces.y"+forces.y+"vely"+vely);
  }
  //Sp = SparsePressure(test);
  //dat.addData(test.t, test.foil.pressForce(test.flow.p), test.foil, test.flow.p);
  //dat.addDataSimple(test.t, test.foil.pressForce(test.flow.p));
  dat.output.println(test.t+" " + test.foil.pressForce(test.flow.p).x+ " "+test.foil.pressForce(test.flow.p).y +" "+test.foil.phi+" "+phivel+" "+ y +" "+vely+" "+ Eta +" "+ CP +" "+ M +" "+ test.foil.dphi +" "+ angle +";");//" "+ Sp +";");
  
  if(test.t<maxT/20*32){
    test.display();
    picNum--;
    if(picNum <= 0){
      saveFrame(datapath + "/" +"frame-#######.png");
      picNum = 30;
    }
  }
  
  
  if (test.t>=maxT/20*20 || testcount==1) {
    dat.finish();
   if (test.t>=maxT/20*20 ){
      // 写入结果, 更改flag;
    PrintWriter output_average1=createWriter("dataY.txt");
    //output_average1.println(CT/count+","+CT/count*2*resolution/(CP/count));
    output_average1.println(CT/count+","+CL/count);
    output_average1.close();
    PrintWriter flg=createWriter("flag.txt");
    flg.println("1");//跑完过后把flag改成1
    flg.close();
    
    //print("test count"+testcount);
   }
   else{
   testcount+=1;
   }
    //更改参数
    BufferedReader input_flag = createReader("flag.txt");
    String content;
    try {
      content = input_flag.readLine();
    } catch (IOException e) {
      e.printStackTrace();
      content = null;
    }
    String[] pieces = split(content, TAB);
    int flagGP = int(pieces[0]);
    println("This is the flagGP : "+ flagGP);
  
  
  
  while(flagGP==1){
      //
      input_flag = createReader("flag.txt");
      try {
        content = input_flag.readLine();
      } catch (IOException e) {
        e.printStackTrace();
        content = null;
      }
      pieces = split(content, TAB);
      flagGP = int(pieces[0]);
      println("This is flagGP : "+ flagGP);
      //
      delay(500);
    };///
  
     BufferedReader input_newParam = createReader("dataX.txt");
    String content1;
    try {
      content1 = input_newParam.readLine();
    } catch (IOException e) {
      e.printStackTrace();
      content1 = null;
    }
    String[] pieces1 = split(content1, ",");
    St = float(pieces1[0]);AD = float(pieces1[1]);
    theta0 = float(pieces1[2]);Phi = float(pieces1[3]);nacaName1 = float(pieces1[4]);nacaName2 = float(pieces1[5]);nacaName3 = float(pieces1[6]);  Re = int(pieces1[7]); // 添加新的雷诺数参数 <-- New change here
    
datapath = "Re"+str(Re)+"/" + str(nacaName1) + str(nacaName2) + str(nacaName3) + '_' + "St=" + str(St) + "_AD=" + str(AD) + "_theta=" + str(theta0) + "_Phi=" + str(Phi); 
       CT = 0; CL = 0; CP = 0;Eta=0;count=0; 
    setUpNewSim();  
  }
}


void setUpNewSim(){
  //创建新的simulation
  //float AoA = 5.0*runNum;
  
  //new File(datapath + "Theta" +str(theta0) + "AD" + str(AD) + "_" + str(Stnum) + "_" + str(Phinum)).mkdir();
  float dAoA = theta0*PI/180., uAoA = dAoA;
  float dA = AD*resolution, uA = dA;
  float phi = Phi*PI/180.;
  float st = St;
  if(St < 0.8){
    maxT = chord/0.3*nflaps;
  }
  else{
  maxT = chord/St*nflaps;
  }
  test = new RotatingFoil(resolution, xLengths, yLengths, tStep, nacaName1,nacaName2,nacaName3, Re, true);
  test.setFlapParams(st,dAoA,uAoA,dA,uA,phi);
  dat = new SaveData(datapath + "/force.txt", test.foil.coords, resolution, xLengths, yLengths, zoom);
    dat.output.println("t"+" "+"fx"+" "+"fy"+" "+"theta"+" "+"thetavel"+" "+"y"+" "+"yvel"+" "+"Eta"+" "+"CP"+" "+"M"+" "+"Dphi"+" "+"Angle");
}
