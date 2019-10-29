/**
*   @author : koseng (Lintang)
*   @brief : Simple implementation of Histogram and Oriented Gradient
*/

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/ml/ml.hpp>

#include <iostream>
#include <sstream>
#include <algorithm>
#include <string>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <dirent.h>

//-------------------------
//-- Select one
#define AMBIL_DATA 0
#define EKSTRAK_FITUR 0
#define TRAINING 1
#define BOOST 0
//-------------------------
#define VISUAL false

using namespace cv;
using namespace std;

vector<vector<float > > all_posdesc;
vector<vector<float > > all_negdesc;

vector<vector<int > > grad_lut(256,vector<int > (256,0));
vector<vector<float > > magn_per_20_lut(1021,vector<float > (1021,0));
vector<vector<float > > angl_lut(2041,vector<float > (2041,0));

void preProses(){    
    for(size_t i=0;i<grad_lut.size();i++){
        for(size_t j=0;j<grad_lut[0].size();j++){
            grad_lut[i][j] = (j-i);
        }
    }
    for(size_t i=0;i<magn_per_20_lut.size();i++){
        for(size_t j=0;j<magn_per_20_lut[0].size();j++){
            magn_per_20_lut[i][j] = sqrt((i*i)+(j*j))/20.0;
        }
    }    

    float kons = 180.0/CV_PI;
    for(size_t i=0;i<angl_lut.size();i++){
        int tmp1 = i;
        if(tmp1>1020)tmp1 = 1020 - i;
        for(size_t j=0;j<angl_lut[0].size();j++){
            int tmp2 = j;
            if(tmp2>1020)tmp2 = 1020 - j;
            float tmp3 = kons*atan2(tmp1,tmp2);
            if(tmp3<0.0)tmp3 = 180 + tmp3;
            angl_lut[i][j] = tmp3;            
        }
    }
}

//=====Gradien pixel pake filter sobel=====//

/*
 * Hasil konvolusi GX sama Input
 * (-1*b1[x-1]) + (0*b1[x]) + (1*b1[x+1]) +
 * (-2*b2[x-1]) + (0*b2[x]) + (2*b2[x+1]) +
 * (-1*b3[x-1]) + (0*b3[x]) + (1*b3[x+1]);
*/
int gradienX(Mat &input,int x, int y){
    int temp;
    uchar *b2 = input.ptr<uchar>(y);
    temp = 0*b2[x];//formalitas
    if(x-1>=0)temp += (-2*b2[x-1]);
    if(x+1<input.cols)temp += 2*b2[x+1];
    if(y-1>=0){
        uchar *b1 = input.ptr<uchar>(y-1);
        temp += (0*b1[x]);//formalitas
        if(x-1>=0)temp += (-1*b1[x-1]);
        if(x+1<input.cols)temp += 1*b1[x+1];
    }
    if(y+1<input.rows){
        uchar *b3 = input.ptr<uchar>(y+1);
        temp += (0*b3[x]);//formalitas
        if(x-1>=0)temp += (-1*b3[x-1]);
        if(x+1<input.cols)temp += 1*b3[x+1];
    }

    return temp;
}

/*
 * Hasil konvolusi GY sama Input
 * (-1*b1[x-1]) + (-2*b1[x]) + (-1*b1[x+1]) +
 * (0*b2[x-1]) + (0*b2[x]) + (0*b2[x+1]) +
 * (1*b3[x-1]) + (2*b3[x]) + (1*b3[x+1]);
*/

int gradienY(Mat &input,int x,int y){
    int temp;
    uchar *b2 = input.ptr<uchar>(y);
    temp = 0*b2[x];//formalitas
    if(x-1>=0)temp += 0*b2[x-1];//formalitas
    if(x+1<input.cols)temp += 0*b2[x+1];//formalitas
    if(y-1>=0){
        uchar *b1 = input.ptr<uchar>(y-1);
        temp += (-2*b1[x]);
        if(x-1>=0)temp += (-1*b1[x-1]);
        if(x+1<input.cols)temp += (-1*b1[x+1]);
    }
    if(y+1<input.rows){
        uchar *b3 = input.ptr<uchar>(y+1);
        temp += (2*b3[x]);
        if(x-1>=0)temp += (1*b3[x-1]);
        if(x+1<input.cols)temp += (1*b3[x+1]);
    }

    return temp;
}

//Standar filter rekomendasi dari HOG(1D Centered)
int gradienX_2(uchar *b2,Mat &input,int x){
    int temp=0;

    if(x-1>=0)temp += b2[x-1];
    if(x+1<input.cols)temp -= b2[x+1];

    return temp;
}

int gradienY_2(uchar *b1,uchar *b3,Mat &input,int x,int y){
    int temp=0;

    if(y+1<input.rows)temp -= b3[x];
    if(y-1>=0)temp += b1[x];

    return temp;
}
//Scharr filter
int gradienX_3(Mat &input,int x, int y){
    int temp;
    uchar *b2 = input.ptr<uchar>(y);
    temp = 0*b2[x];//formalitas
    if(x-1>=0)temp += (-10*b2[x-1]);
    if(x+1<input.cols)temp += 10*b2[x+1];
    if(y-1>=0){
        uchar *b1 = input.ptr<uchar>(y-1);
        temp += (0*b1[x]);//formalitas
        if(x-1>=0)temp += (-3*b1[x-1]);
        if(x+1<input.cols)temp += 3*b1[x+1];
    }
    if(y+1<input.rows){
        uchar *b3 = input.ptr<uchar>(y+1);
        temp += (0*b3[x]);//formalitas
        if(x-1>=0)temp += (-3*b3[x-1]);
        if(x+1<input.cols)temp += 3*b3[x+1];
    }

    return temp;
}

int gradienY_3(Mat &input,int x,int y){
    int temp;
    uchar *b2 = input.ptr<uchar>(y);
    temp = 0*b2[x];//formalitas
    if(x-1>=0)temp += 0*b2[x-1];//formalitas
    if(x+1<input.cols)temp += 0*b2[x+1];//formalitas
    if(y-1>=0){
        uchar *b1 = input.ptr<uchar>(y-1);
        temp += (-10*b1[x]);
        if(x-1>=0)temp += (-3*b1[x-1]);
        if(x+1<input.cols)temp += (-3*b1[x+1]);
    }
    if(y+1<input.rows){
        uchar *b3 = input.ptr<uchar>(y+1);
        temp += (10*b3[x]);
        if(x-1>=0)temp += (3*b3[x-1]);
        if(x+1<input.cols)temp += (3*b3[x+1]);
    }

    return temp;
}

void ekstrak_fitur_HOG(Mat img,Size win_size, int cell_size,int block_size, float overlap, int nbins,vector<float> &desc,bool visualisasi){
    Mat gray;
    int pos_x=0,pos_y=0;
    int inc = (float)block_size*overlap;

    vector<float > quad_sum;
    vector<vector<float > > histcell;
    //vector<vector<float > > normblock;

    if(img.channels()!=1){
        cvtColor(img,gray,CV_BGR2GRAY);
    }else{
        img.copyTo(gray);
    }

    if(gray.cols!=win_size.width||gray.rows!=win_size.height){
        resize(gray,gray,win_size);
    }    

//===============================
    Mat visual = gray.clone();
    Mat hog_space;
    int pos_x_visual=0;
    int pos_y_visual=0;
    int skala=5;
    if(visualisasi){
        resize(visual,visual,Size(win_size.width*skala,win_size.height*skala));
        hog_space = Mat::zeros(visual.size(),CV_8UC1);
        cvtColor(visual,visual,CV_GRAY2BGR);
    }
//===============================

    Mat cek_aj = Mat(gray.size(),CV_32FC1);

    while(pos_y<win_size.height){

        vector<float > vtemp(nbins);

        float temp=0;        

        for(int i=0;i<cell_size;i++){

//            uchar *b1,*b2,*b3;
//            if(i-1>=0)b1 = mcell.data+mcell.step*(i-1);
//            if(i+1<mcell.rows)b3 = mcell.data+mcell.step*(i+1);
//            b2 = mcell.data+mcell.step*i;

            for(int j=0;j<cell_size;j++){

                int gx = gradienX(gray,pos_x + j, pos_y + i);
                int gy = gradienY(gray,pos_x + j, pos_y + i);

//                int gx = grad_lut[(j+1<mcell.cols)?b2[j+1]:0][(j-1<0)?0:b2[j-1]];
//                int gy = grad_lut[(i+1<mcell.rows)?b3[j]:0][(i-1<0)?0:b1[j]];

                float mag = magn_per_20_lut[abs(gx)][abs(gy)];

                cek_aj.at<float > (pos_y + i,pos_x + j) = mag;

                float ang = angl_lut[(gy<0)?1020-gy:gy][(gx<0)?1020-gx:gx];

                int interval = ang/20.0;
                switch(interval){
                    case 0:vtemp[0] += (20.0-ang)*mag; vtemp[1] += ang*mag;break;
                    case 1:vtemp[1] += (40.0-ang)*mag; vtemp[2] += (ang-20.0)*mag;break;
                    case 2:vtemp[2] += (60.0-ang)*mag; vtemp[3] += (ang-40.0)*mag;break;
                    case 3:vtemp[3] += (80.0-ang)*mag; vtemp[4] += (ang-60.0)*mag;break;
                    case 4:vtemp[4] += (100.0-ang)*mag; vtemp[5] += (ang-80.0)*mag;break;
                    case 5:vtemp[5] += (120.0-ang)*mag; vtemp[6] += (ang-100.0)*mag;break;
                    case 6:vtemp[6] += (140.0-ang)*mag; vtemp[7] += (ang-120.0)*mag;break;
                    case 7:vtemp[7] += (160.0-ang)*mag; vtemp[8] += (ang-140.0)*mag;break;
                    default:vtemp[8] += (180.0-ang)*mag; vtemp[0] += (ang-160.0)*mag;break;
                }
             }
        }       

        histcell.push_back(vtemp);

        float maks_hist=0.0;

        for(vector<float>::iterator it=vtemp.begin();it!=vtemp.end();++it){

            if(*it > maks_hist){
                maks_hist =  *it;
            }

            temp += (*it) * (*it);
        }        

        quad_sum.push_back(temp);


        if(visualisasi){
            Rect cell=Rect(pos_x_visual,pos_y_visual,skala*cell_size,skala*cell_size);

            Mat visual_roi = Mat(visual,cell);
            Mat hog_space_roi = Mat(hog_space,cell);

            for(int i=0;i<nbins;i++){
                float panjang = (vtemp[i]/maks_hist)*cell_size*skala;
                int kons = (cell_size*skala)/2;
                int x1 = kons - panjang/2;
                int x2 = kons + panjang/2;
                int y1 = kons + tan(((double)(i*20) * CV_PI) / 180.0)*(x1-kons);
                int y2 = kons + tan(((double)(i*20) * CV_PI) / 180.0)*(x2-kons);

                line(visual_roi,Point(x1,y1),Point(x2,y2),Scalar(0,0,255),1);

                line(hog_space_roi,Point(x1,y1),Point(x2,y2),Scalar(((vtemp[i]*255.0)/maks_hist)),1);
            }

            rectangle(visual,cell,Scalar(255,0,0),1);
        }

        pos_x+=cell_size;        


        pos_x_visual+=cell_size*skala;

        if(pos_x>=win_size.width){
            pos_x=0;
            pos_y+=cell_size;            

            pos_x_visual=0;
            pos_y_visual+=cell_size*skala;
        }        
    }    

    gray.release();

    pos_x=0;
    pos_y=1;

    //int limit_x = cell_size - 1;

    while(pos_y%cell_size!=0){
        float temp=0;
        vector<float > vtemp;
        vtemp.insert(vtemp.end(),histcell[pos_x].begin(),histcell[pos_x].end());
        vtemp.insert(vtemp.end(),histcell[pos_x+1].begin(),histcell[pos_x+1].end());
        vtemp.insert(vtemp.end(),histcell[pos_x+8].begin(),histcell[pos_x+8].end());
        vtemp.insert(vtemp.end(),histcell[pos_x+9].begin(),histcell[pos_x+9].end());

        temp = quad_sum[pos_x] + quad_sum[pos_x+1]+ quad_sum[pos_x+8] + quad_sum[pos_x+9];

        temp = sqrt(temp);//L2 Norm

        for(vector<float >::iterator it=vtemp.begin();it!=vtemp.end();++it){
            *it /= temp;
        }        

        //normblock.push_back(vtemp);
        //cout<<"Block Size : "<<vtemp.size()<<endl;
        desc.insert(desc.end(),vtemp.begin(),vtemp.end());
        pos_x += inc;
        if(pos_x%(cell_size*pos_y - 1)== 0){
            pos_x+=inc;
            pos_y+=inc;
            //cout<<pos_x<<" ; "<<(pos_y-1)<<endl;
        }
    }

    if(visualisasi){

        normalize(cek_aj,cek_aj,0,1,NORM_MINMAX);

        imshow("GRAD",cek_aj);
        imshow("HS",hog_space);
        namedWindow("VISUAL",CV_WINDOW_NORMAL);
        imshow("VISUAL",visual);

        waitKey(0);
    }
}

//============================================================

int uk_cuplikan=500;
//============================================================
bool klik_mouse=false;
bool imshow_flag=false;
bool klik_kanan=false;
int tlx=0,tly=0;
int brx=0,bry=0;
int mouse_x=0,mouse_y=0;

void callBack(int event,int x,int y,int flags,void *userdata){
    if(event == EVENT_LBUTTONDOWN){
        if(!klik_mouse){
            tlx=x;
            tly=y;
        }else{
            brx=x;
            bry=y;
            if(brx-tlx>uk_cuplikan)brx=tlx+uk_cuplikan;
            else if(brx-tlx<-uk_cuplikan)brx=tlx-uk_cuplikan;
            if(bry-tly>uk_cuplikan)bry=tly+uk_cuplikan;
            else if(bry-tly<-uk_cuplikan)bry=tly-uk_cuplikan;
            imshow_flag=true;
        }
        klik_mouse = !klik_mouse;
    }else if(event == EVENT_MOUSEMOVE){
        mouse_x=x;
        mouse_y=y;
    }else if(event == EVENT_RBUTTONDOWN){
        klik_kanan=true;
        tlx=x;
        tly=y;
    }
}
//============================================================


int main(){
    stringstream dir_posfile;
    dir_posfile << "/mnt/E/alfarobi/data/bola/HOG/positive/";
    stringstream dir_negfile;
    dir_negfile << "/mnt/E/alfarobi/data/bola/HOG/negative/";
    stringstream dir_hasiltraining;
    dir_hasiltraining << "/mnt/E/alfarobi/data/bola/HOG/bola.xml";

#if EKSTRAK_FITUR==1 || AMBIL_DATA==1
    DIR *dir;
    struct dirent *ent;
#endif   


#if AMBIL_DATA==1

    // uncomment misal pakai kamera
//    int set_kamera = system("v4l2-ctl --device=1 --set-ctrl exposure_auto_priority=0,exposure_auto=1,exposure_absolute=300,gain=200,brightness=128,contrast=150,saturation=200,focus_auto=0,white_balance_temperature_auto=0,white_balance_temperature=5500");
//    set_kamera = system("v4l2-ctl --device=1 --set-ctrl exposure_auto_priority=0,exposure_auto=1,exposure_absolute=300,gain=200,brightness=128,contrast=150,saturation=200,focus_auto=0,white_balance_temperature_auto=0,white_balance_temperature=5500");
//    cout<<"Respon Kamera : "<<set_kamera<<endl;

    VideoCapture vc("/mnt/E/alfarobi/video9.avi");
    //VideoCapture vc("/media/koseng/563C3F913C3F6ADF/backup/hasil_rekam/video2.avi");
    //VideoCapture vc("/media/koseng/563C3F913C3F6ADF/bola/video_/Robots playing soccer at RoboCup 2015 is like watching toddlers learn to kick - Mashable.mp4");
    //VideoCapture vc("/home/koseng/Downloads/MRL-HSL Qualification Video for RoboCup 2018 humanoid KidSize.mp4");
    //VideoCapture vc("/media/koseng/563C3F913C3F6ADF/Photos/ikut/RoboCup 2017 NimbRo TeenSize Round Robin 480p.mp4");
    //VideoCapture vc(1);
    //VideoCapture vc("/media/koseng/563C3F913C3F6ADF/tesss.avi");

    char c;
    cout<<"Masukkan 'y' untuk memulai : ";
    cin >> c;
    if(c!='y')exit(0);
    cout.flush();

    int banyak_file=-2;

    if((dir=opendir(dir_posfile.str().c_str()))!=NULL){
        while((ent = readdir(dir))!=NULL){
            banyak_file++;
        }
        cout<<"Jumlah file positif saat ini : "<<banyak_file<<endl;
    }else{
        cerr<<"Direktori gambar positif tidak ditemukan !!"<<endl;
        exit(0);
    }

    banyak_file=-2;
    if((dir=opendir(dir_negfile.str().c_str()))!=NULL){
        while((ent = readdir(dir))!=NULL){
            banyak_file++;
        }
        cout<<"Jumlah file negatif saat ini : "<<banyak_file<<endl;
    }else{
        cerr<<"Direktori gambar negatif tidak ditemukan !!"<<endl;
        exit(0);
    }


    namedWindow("FRAME",CV_WINDOW_NORMAL);
    namedWindow("CEK",CV_WINDOW_NORMAL);
    while(1){
        Mat frame,frame2;
        vc >> frame;
//========Cek Kernel========
//        Mat gray;
//        cvtColor(frame,gray,CV_BGR2GRAY);
//        Mat gx(frame.size(),CV_32FC1);
//        Mat gy(frame.size(),CV_32FC1);
//        for(int i=0;i<frame.rows;i++){
//            float* gx_ptr = gx.ptr<float>(i);
//            float* gy_ptr = gy.ptr<float>(i);
//            for(int j=0;j<frame.cols;j++){
//                gx_ptr[j] = gradienX(gray,j,i);
//                gy_ptr[j] = gradienY(gray,j,i);
//            }
//        }
//        normalize(gx,gx,0,1,NORM_MINMAX);
//        normalize(gy,gy,0,1,NORM_MINMAX);

//        imshow("GX",gx);
//        imshow("GY",gy);

        //resize(frame,frame,Size(320,240));
        //flip(frame,frame,-1);
        frame.copyTo(frame2);
        //cvtColor(frame,frame,CV_BGR2GRAY);
        imshow("FRAME",frame2);
        int c = waitKey(30);
        if(c==32){
            namedWindow("CEK",CV_WINDOW_AUTOSIZE);
            setMouseCallback("CEK",callBack,NULL);
            while(1){
                Mat copy;
                frame.copyTo(copy);
                Rect kotak;
                if(klik_mouse){
                    if(mouse_y-tly>uk_cuplikan)mouse_y=tly+uk_cuplikan;
                    else if(mouse_y-tly<-uk_cuplikan)mouse_y=tly-uk_cuplikan;
                    if(mouse_x-tlx>uk_cuplikan)mouse_x=tlx+uk_cuplikan;
                    else if(mouse_x-tlx<-uk_cuplikan)mouse_x=tlx-uk_cuplikan;
                    //int pos_x = tlx
                    int panjang_sisi=min(abs(tlx-mouse_x),abs(tly-mouse_y));
                    kotak = Rect(tlx,tly,panjang_sisi,panjang_sisi);
                    rectangle(copy,kotak,Scalar(0,0,255),1);
//                    line(copy,Point(tlx,tly),Point(mouse_x,tly),Scalar(0,0,255),1);
//                    line(copy,Point(tlx,tly),Point(tlx,mouse_y),Scalar(0,0,255),1);
//                    line(copy,Point(mouse_x,tly),Point(mouse_x,mouse_y),Scalar(0,0,255),1);
//                    line(copy,Point(tlx,mouse_y),Point(mouse_x,mouse_y),Scalar(0,0,255),1);
                }
                imshow("CEK",copy);
                int ch = waitKey(10);
                if(imshow_flag){
                    imshow_flag=false;
                    int temp_tlx=tlx;
                    int temp_tly=tly;
                    if(tlx>brx)temp_tlx = brx;
                    if(tly>bry)temp_tly = bry;
                    Rect r(temp_tlx,temp_tly,abs(tlx-brx),abs(tlx-brx));
                    Mat roi(frame,kotak);
                    imshow("ROI",roi);
                    while(1){
                        int ch2=waitKey(0)&0xFF;
                        if(ch2==27){cout<<"Gambar tidak disimpan"<<endl;break;}
                        else{
                            stringstream ss;
                            time_t saat_ini;
                            time(&saat_ini);
                            string namafile(ctime(&saat_ini));
                            namafile.erase(namafile.end()-1);
                            replace(namafile.begin(),namafile.end(),' ','_');
                            cout<<namafile<<endl;
                            if(ch2==112){
                                ss<<dir_posfile.str()<<namafile<<"_pos.jpg";
                                imwrite(ss.str(),roi);
                                cout<<"Gambar Positif disimpan !!"<<endl;
                                break;
                            }else if(ch2==110){
                                ss<<dir_negfile.str()<<namafile<<"_neg.jpg";
                                imwrite(ss.str(),roi);
                                cout<<"Gambar Negatif disimpan !!"<<endl;
                                break;
                            }
                        }
                    }
                }else if(ch==27){
                    if(klik_mouse){
                        klik_mouse=false;
                    }else{
                        break;
                    }
                }
            }
        }else if(c==27){
            break;
        }else{
            continue;
        }
    }

#endif

#if EKSTRAK_FITUR==1

    preProses();
    Size win_size(64,64);//piksel
    int cell_size=8;//piksel
    int block_size=2;//sel
    int nbins=9;//jumlah bin
    float overlap = 0.5;//blok

    cout<<"Eks Pos....."<<endl;    
    if((dir=opendir(dir_posfile.str().c_str()))!=NULL){
        while((ent = readdir(dir))!=NULL){
            char buff[100];
            sprintf(buff,"%s%s",dir_posfile.str().c_str(),ent->d_name);
            Mat input = imread(buff);
            if(input.data){


//                Mat splt[3];
//                split(input,splt);
//                Mat roi_gray = splt[2].clone();
//                Mat filter;

//                GaussianBlur(roi_gray,filter,Size(0,0),3);
//                addWeighted(roi_gray,3.0,filter,-1.5,0,filter);

                vector<float > desc;
                ekstrak_fitur_HOG(input,win_size,cell_size,block_size,overlap,nbins,desc,VISUAL);
                if(desc.size()){
                    all_posdesc.push_back(desc);
                }
            }            
        }
    }else{
        cerr<<"Direktori tidak ditemukan !!"<<endl;
    }    
    FileStorage hdx_pos("/mnt/E/alfarobi/data/bola/HOG/positive.xml",FileStorage::WRITE);
    int jml_pos_data = all_posdesc.size();
    int desc_size = all_posdesc[0].size();
    Mat M_pos(jml_pos_data,desc_size,CV_32F);
    cout<<"Simpan...."<<endl;
    for(int i=0;i<jml_pos_data;i++)
        memcpy(&(M_pos.data[desc_size*i*sizeof(float)]),all_posdesc[i].data(),desc_size*sizeof(float));
    write(hdx_pos,"Pos_Descriptor",M_pos);
    hdx_pos.release();
    cout<<"Ekstrak fitur dari gambar positif selesai !!!"<<endl;

    cout<<"Eks Neg....."<<endl;
    if((dir=opendir(dir_negfile.str().c_str()))!=NULL){        
        while((ent = readdir(dir))!=NULL){
            char buff[100];
            sprintf(buff,"%s%s",dir_negfile.str().c_str(),ent->d_name);
            Mat input = imread(buff);

            if(input.data){
//                imshow("IN",input);

//                if(input.cols*input.rows <= win_size.area()){
//                    resize(input,input,Size(max(win_size.width,input.cols),max(win_size.height,input.rows)));
//                }
                resize(input,input,win_size);
                int x=0,y=0;
                int oob_x = input.cols - win_size.width;
                int oob_y = input.rows - win_size.height;
                while(1){
//                    Mat temp;
//                    input.copyTo(temp);
//                    Rect roi(x,y,win_size.width,win_size.height);
//                    Mat mroi(input,roi);

//                    Mat splt[3];
//                    split(input,splt);
//                    Mat roi_gray = splt[2].clone();
//                    Mat filter;

//                    GaussianBlur(roi_gray,filter,Size(0,0),3);
//                    addWeighted(roi_gray,3.0,filter,-1.5,0,filter);

                    vector<float > desc;
                    ekstrak_fitur_HOG(input,win_size,cell_size,block_size,overlap,nbins,desc,VISUAL);
                    if(desc.size()){
                        all_negdesc.push_back(desc);
                    }

//                    rectangle(temp,roi,Scalar(0,255,0),2);

                    x+=win_size.width;

                    if(x==input.cols){
                        x=0;
                        y+=win_size.height;
                        if(y==input.cols)break;
                        else if(y>oob_y)y=oob_y;
                    }else if(x>oob_x){
                        x = oob_x;
                    }
                }
            }            
        }
    }else{
        cerr<<"Direktori tidak ditemukan !!"<<endl;
    }
    FileStorage hdx_neg("/mnt/E/alfarobi/data/bola/HOG/negative.xml",FileStorage::WRITE);
    int jml_neg_data = all_negdesc.size();
    Mat M_neg(jml_neg_data,desc_size,CV_32F);
    cout<<"Simpan...."<<endl;
    for(int i=0;i<jml_neg_data;i++)
        memcpy(&(M_neg.data[desc_size*i*sizeof(float)]),all_negdesc[i].data(),desc_size*sizeof(float));
    write(hdx_neg,"Neg_Descriptor",M_neg);
    hdx_neg.release();
    cout<<"Ekstrak fitur dari gambar negatif selesai !!!"<<endl;
#endif

#if TRAINING == 1
    FileStorage read_pos;
    read_pos.open("/mnt/E/alfarobi/data/bola/HOG/positive.xml",FileStorage::READ);
    Mat M_pos;
    read_pos["Pos_Descriptor"] >> M_pos;
    int jml_pos_data=M_pos.rows,desc_size=M_pos.cols;
    read_pos.release();

    FileStorage read_neg;
    read_neg.open("/mnt/E/alfarobi/data/bola/HOG/negative.xml",FileStorage::READ);
    Mat M_neg;
    read_neg["Neg_Descriptor"] >> M_neg;
    int jml_neg_data=M_neg.rows;
    read_neg.release();

    Mat train_data(jml_pos_data + jml_neg_data,desc_size,CV_32FC1);

    //Mat train_data(jml_pos_data,desc_size,CV_32FC1);//ONE CL

    int kons = sizeof(float)*jml_pos_data*desc_size;

    memcpy(train_data.data,M_pos.data,kons);

    memcpy(&train_data.data[kons],M_neg.data,sizeof(float)*jml_neg_data*desc_size);

    Mat labels(jml_pos_data + jml_neg_data,1,CV_32SC1,Scalar(-1));

    //Mat labels(jml_pos_data,1,CV_32FC1,Scalar(1));//ONE CL

#if BOOST==1
    labels.rowRange(jml_pos_data + 1, jml_pos_data + jml_neg_data) = Scalar(0);

    CvBoostParams params;
    params.boost_type = CvBoost::REAL;
    params.min_sample_count = 5;
    params.weak_count = 100;
    params.weight_trim_rate = 0;
    params.cv_folds = 0;
    params.max_depth = 5;
    params.truncate_pruned_tree = false;
    params.regression_accuracy = 0.0;
    params.use_surrogates=false;
    CvBoost boost;

    boost.train(train_data,CV_ROW_SAMPLE,labels,Mat(),Mat(),Mat(),Mat(),params);
    boost.save(dir_hasiltraining.str().c_str());

#else
    cout<<"Training mulai !!"<<endl;

    labels.rowRange(0, jml_pos_data) = Scalar(1);    
    Ptr<ml::SVM> svm = ml::SVM::create();
    svm->setNu(0.01);
    svm->setDegree(3);
    svm->setType(ml::SVM::NU_SVC);
    svm->setKernel(ml::SVM::LINEAR);
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER,10000,0));

//    CvSVMParams params;
    //params.C=0.1;
    //params.p=0.1;

//    params.nu=0.01;

    //params.coef0=0.0;

//    params.degree=3;

    //params.gamma=0;

//    params.svm_type = CvSVM::NU_SVC;
//    params.kernel_type = CvSVM::LINEAR;

    //params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 10000, 1e-6);

//    params.term_crit = TermCriteria(TermCriteria::MAX_ITER,10000,0);

//    svm.train_auto(train_data,labels,Mat(),Mat(),params);
    Ptr<ml::TrainData> td = ml::TrainData::create(train_data,ml::SampleTypes::ROW_SAMPLE,labels);
//    svm->trainAuto(td);
    svm->train(td);
    //svm.train(train_data,labels,Mat(),Mat(),params);
//    svm.save(dir_hasiltraining.str().c_str());
    svm->save(dir_hasiltraining.str().c_str());
    cout<<"Training beress!!"<<endl;

#endif
#endif


    return 0;
}
