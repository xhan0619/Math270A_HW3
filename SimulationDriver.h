#ifndef _SIMULATION_DRIVER_H_
#define _SIMULATION_DRIVER_H_

#include <Eigen/Dense>
#include <Eigen/Core>
#include <fstream>
#include "LagrangianForce.h"
#include "cassert"

namespace FILE_IO{
  inline void Write_Binary(std::string directory, std::string name,Eigen::VectorXd& v){
    std::string filename(directory+std::string("/")+name+std::string(".binary"));
    std::ofstream outdata(filename.c_str(),std::ios::out|std::ios::binary);
    for(int i=0;i<v.size();i++)
      outdata.write(reinterpret_cast<char*>(&v(i)),sizeof(double));
    outdata.close();
  }

  inline void Read_Binary(std::string directory, std::string name,Eigen::VectorXd& v){
    std::string filename(directory+std::string("/")+name+std::string(".binary"));
    std::ifstream indata(filename.c_str(),std::ios::in|std::ios::binary);
    for(int i=0;i<v.size();i++)
      indata.read(reinterpret_cast<char*>(&v(i)),sizeof(double));
    indata.close();
  }

  inline void Write_Binary(std::string directory, std::string name,Eigen::VectorXf& v){
    std::string filename(directory+std::string("/")+name+std::string(".binary"));
    std::ofstream outdata(filename.c_str(),std::ios::out|std::ios::binary);
    for(int i=0;i<v.size();i++)
      outdata.write(reinterpret_cast<char*>(&v(i)),sizeof(float));
    outdata.close();
  }

  inline void Read_Binary(std::string directory, std::string name,Eigen::VectorXf& v){
    std::string filename(directory+std::string("/")+name+std::string(".binary"));
    std::ifstream indata(filename.c_str(),std::ios::in|std::ios::binary);
    for(int i=0;i<v.size();i++)
      indata.read(reinterpret_cast<char*>(&v(i)),sizeof(float));
    indata.close();
  }

  inline void Write_DAT_File(std::string file,const Eigen::VectorXf& array){
    FILE* fpointer;
    fpointer=fopen(file.c_str(),"w");
    for(int i=0;i<array.size();i++)
        fprintf(fpointer,"%f\n",array(i));
    fclose(fpointer);
}

inline void Write_DAT_File(std::string file,const Eigen::VectorXd& array){
    FILE* fpointer;
    fpointer=fopen(file.c_str(),"w");
    for(int i=0;i<array.size();i++)
        fprintf(fpointer,"%g\n",array(i));
    fclose(fpointer);
}
}

namespace JIXIE{

template <class T>
class SimulationParameters{
public:
  T final_time;
  int frames_per_second;
  T dt;
  std::string output_dir;
  SimulationParameters(){}
};

template <class T>
class SimulationDriver{
public:
T time,dt,dt_target,dt_frame,final_time,dt_min;
int frames_per_second,current_frame;
std::string output_directory;

  SimulationDriver(const T final_time_input,const int frames_per_second_input,const T dt_input,std::string& output_dir):
  final_time(final_time_input),frames_per_second(frames_per_second_input),dt(dt_input),dt_target(dt),
  dt_min((T)1e-6),output_directory(output_dir){
    dt_frame=(T)1/(T)frames_per_second;
    if(dt_target > dt_frame)
      dt_target=dt_frame;
    dt=dt_target;
  }

  SimulationDriver(SimulationParameters<T>& parameters):
  SimulationDriver(parameters.final_time,parameters.frames_per_second,parameters.dt,parameters.output_dir){}

  virtual void Set_Dt(bool& write_frame){
    dt=dt_target;
    if((T)(current_frame + 1) * dt_frame - time < dt_min + dt){
      dt=(T)(current_frame + 1) * dt_frame - time;
      current_frame++;
      write_frame=true;
    }
    else if(time + dt > (T)(current_frame + 1) * dt_frame){
      dt=(T)(current_frame + 1) * dt_frame - time;
      current_frame++;
      write_frame=true;
    }
    else write_frame=false;
  }

  virtual void Initialize(){
    time=(T)0;
    WriteState(current_frame);
  }

  virtual void Write_Obj(const int number){}

  void WriteState(const int number){
    std::ofstream outdata;
    std::string simulation_data_filename(output_directory+std::string("/simulation_info.dat"));
    outdata.open(simulation_data_filename.c_str());
    outdata << number << std::endl;
    outdata.close();
    Write_State(number,simulation_data_filename);
  }

  virtual void Write_State(const int number,std::string& simulation_data_filename){}
  virtual bool Read_State(const int number,std::string& simulation_data_filename){return false;}

  virtual void Advance_One_Time_Step(const bool verbose){
    time+=dt;
  }

  void RunSimulation(const bool verbose=false){

    Initialize();
    while(time<final_time){
      if(verbose)
        std::cout << "Time = " << time << ", frame = " << current_frame << ", dt = " << dt << std::endl;
      bool write_to_file=false;
      Set_Dt(write_to_file);
      Advance_One_Time_Step(verbose);
      if(write_to_file){
        WriteState(current_frame);
        Write_Obj(current_frame);
      }
    }
  }
};

template <class T>
class ElasticityParameters:public SimulationParameters<T>{
public:
  int N;
  T a;
  T dX;
  T rho;
  T k;
  T Newton_tol;
  T tb; // XH: added right side Neumann BC
  T width; // XH: added parameter width
  int max_newton_it;

  ElasticityParameters(){}
};

template <class T>
class ElasticityDriver: public SimulationDriver<T>{
  using SimulationDriver<T>::output_directory;
  using SimulationDriver<T>::time;
  using SimulationDriver<T>::dt;
  typedef Eigen::Matrix<T,Eigen::Dynamic, 1> TVect;
  int N;
  T a,dX;
  T rho,k;
  T tb; // XH added right side Neumann BC
  T width; // XH: twice the initial width of the cross section of the jello TODO: add width to parameters class
  TVect x_n, x_np1,v_n,x_hat,residual,mass,delta;
  TVect xyz_n; // XH: record the location of all points in 3D
  TVect faces; // XH: record the triangulation meshes
  T Newton_tol;
  int max_newton_it;
  ConstitutiveModel<T>* cons_model;
  LagrangianForces<T>* lf;
  SymmetricTridiagonal<T> be_matrix;
public:

  ElasticityDriver(ElasticityParameters<T>& parameters):
  SimulationDriver<T>(parameters),N(parameters.N),a(parameters.a),dX(parameters.dX),
  rho(parameters.rho),k(parameters.k),tb(parameters.tb), x_n(parameters.N), xyz_n(12 * parameters.N), x_np1(parameters.N),v_n(parameters.N),x_hat(parameters.N),residual(parameters.N),mass(parameters.N),delta(parameters.N),
  Newton_tol(parameters.Newton_tol),max_newton_it(parameters.max_newton_it),be_matrix(parameters.N), width(parameters.width), faces(24 * (N-1) + 12){
    //cons_model=new LinearElasticity<T>(k);
    cons_model=new NeoHookean<T>(k);
    lf=new FEMHyperelasticity<T>(a,dX,N,*cons_model);
  }

  ~ElasticityDriver(){
    delete cons_model;
    delete lf;
  }
  // void Form_xyz(const TVect x_n, TVect xyz_n){
  //   for (int i = 0; i < N; i++){
  //     for (int j = 0; j < 4; j++){
  //       for (int k = 0; k < 3; k++){
  //         xyz_n(12*i + 4*j + k) = x
  //       }
  //     }
  //   }
  // }
  void Initialize(){
    //set intiial positions and velocity
    for(int i=0;i<N;i++){
      T x=(a+(T)i*dX);
      x_n(i)=(T).7*x;
      v_n(i)=(T)0;
    }

    for (int i = 0; i < N; i++){
      xyz_n(i*12) = x_n(i);
      xyz_n(i*12 + 1) = width;
      xyz_n(i*12 + 2) = width;

      xyz_n(i*12 + 3) = x_n(i);
      xyz_n(i*12 + 4) = -width;
      xyz_n(i*12 + 5) = width;
      
      xyz_n(i*12 + 6) = x_n(i);
      xyz_n(i*12 + 7) = -width;
      xyz_n(i*12 + 8) = -width;
      
      xyz_n(i*12 + 9) = x_n(i);
      xyz_n(i*12 + 10) = width;
      xyz_n(i*12 + 11) = -width;
    }
    //intialize mass lumped mass matrix from density
    for(int e=0;e<N-1;e++){
      mass(e)+=(T).5*rho*dX;

      // XH: fixed the second entry of the mass matrix according to Dirichlet BC on left
      if (e > 0){
          mass(e+1)+=(T).5*rho*dX;
      }
      else {
           mass(e+1)+=((T)1 / (T)3) *rho*dX;
      }
      //assert(xyz_n.size() == 12 * N);
    }

    SimulationDriver<T>::Initialize();

    // XH: initialize faces
    for (int i = 0; i < N-1; i++){
      faces(24 * i + 0) = (4*i+ 1);
      faces(24 * i + 1) = (4*i+ 1)+1;
      faces(24 * i + 2) = (4*i+ 1)+5;

      faces(24 * i + 3) = (4*i+ 1);
      faces(24 * i + 4) = (4*i+ 1)+5;
      faces(24 * i + 5) = (4*i+ 1)+4;

      faces(24 * i + 6) = (4*i+ 1);
      faces(24 * i + 7) = (4*i+ 1)+4;
      faces(24 * i + 8) = (4*i+ 1)+3;

      faces(24 * i + 9) = (4*i+ 1)+3;
      faces(24 * i + 10) = (4*i+ 1)+4;
      faces(24 * i + 11) = (4*i+ 1)+7;

      faces(24 * i + 12) = (4*i+ 1)+2;
      faces(24 * i + 13) = (4*i+ 1)+3;
      faces(24 * i + 14) = (4*i+ 1)+6;

      faces(24 * i + 15) = (4*i+ 1)+3;
      faces(24 * i + 16) = (4*i+ 1)+7;
      faces(24 * i + 17) = (4*i+ 1)+6;

      faces(24 * i + 18) = (4*i+ 1) + 1;
      faces(24 * i + 19) = (4*i+ 1) + 2;
      faces(24 * i + 20) = (4*i+ 1)+5;
      faces(24 * i + 21) = (4*i+ 1)+2;
      faces(24 * i + 22) = (4*i+ 1)+6;
      faces(24 * i + 23) = (4*i+ 1)+5;
    }
    faces(24 * (N-1)) = 1;
    faces(24 * (N-1)+1) = 3;
    faces(24 * (N-1)+2) = 2;

    faces(24 * (N-1)+3) = 1;
    faces(24 * (N-1)+4) = 4;
    faces(24 * (N-1)+5) = 3;

    faces(24 * (N-1)+6) = 4*N;
    faces(24 * (N-1)+7) = 4*N-3;
    faces(24 * (N-1)+8) = 4*N-1;

    faces(24 * (N-1)+9) = 4*N-3;
    faces(24 * (N-1)+10) = 4*N-2;
    faces(24 * (N-1)+11) = 4*N-1;
  }

  // XH added helper function to calculate d\phi / dX
  T F(const TVect& x,const int e)const{
    return (x(e)-x(e-1))/dX;
  }

  virtual void Advance_One_Time_Step(const bool verbose){
    time+=dt;
    x_hat=x_n+dt*v_n;
    x_np1=x_n;//initial guess

    for(int it=1;it<max_newton_it;it++){
      residual=mass.asDiagonal()*(x_hat-x_np1);
      lf->AddForce(residual,x_np1, dt*dt); // XH: Added the right side Neumann BC
      residual(0) = 0; // XH: force left end point residual = 0 because of Dirichlet BC
      residual(N-1) += tb*dt*dt; // // XH: add the force on the right side because of Dirichlet BC
      T norm=(T)0;for(int i=0;i<N;i++) norm+=residual(i)*residual(i)/mass(i);
      norm=sqrt(norm);
      if(verbose)
        std::cout << "Newton residual at iteration " << it << " = " << norm << std::endl;
      if(norm<Newton_tol){
        Exit_BE();
        return;}
      be_matrix.SetToZero();
      for(int i=0;i<N;i++) be_matrix(i,i)=mass(i);
      lf->AddForceDerivative(be_matrix,x_np1,-dt*dt);
      be_matrix.GMRESSolve(delta,residual); //XH: switched to GMRES Solve
      x_np1+=delta;
    }
    Exit_BE();
  }

  void Exit_BE(){
    v_n=((T)1/dt)*(x_np1-x_n);
    x_n=x_np1;
    for (int i = 0; i < N; i++){
      if (i == 0)
        a = JIXIE::MATH_TOOLS::rsqrt(F(x_n,i+1));
      else
        a = JIXIE::MATH_TOOLS::rsqrt(F(x_n,i));
      xyz_n(i*12) = x_n(i);
      xyz_n(i*12 + 1) = width * a;
      xyz_n(i*12 + 2) = width * a;

      xyz_n(i*12 + 3) = x_n(i);
      xyz_n(i*12 + 4) = -width * a;
      xyz_n(i*12 + 5) = width * a;
      
      xyz_n(i*12 + 6) = x_n(i);
      xyz_n(i*12 + 7) = -width * a;
      xyz_n(i*12 + 8) = -width * a;
      
      xyz_n(i*12 + 9) = x_n(i);
      xyz_n(i*12 + 10) = width * a;
      xyz_n(i*12 + 11) = -width * a;
    }
  }

  virtual void Write_Obj(const int number){
    char str[12];
    sprintf(str, "%d", number);
    std::string frame_name(str);
    std::string positions_filename(output_directory + std::string("/ particle_xyz_") + frame_name + ".obj");
    std::ofstream myfile(positions_filename);

    for (int i = 0; i < 4*N; i ++){
      myfile << "v " << xyz_n(3 * i) << " " << xyz_n(3*i + 1) << " " << xyz_n(3*i + 2) << std::endl;
    }
    for (int i = 0; i < 8*(N-1) + 4 ; i ++){
      myfile << "f " << faces(3 * i) << " " << faces(3*i + 1) << " " << faces(3*i + 2) << std::endl;
    }
    myfile.close();
  }

  void Write_State(const int number,std::string& simulation_data_filename){
    std::ofstream basic_outdata;
    basic_outdata.open(simulation_data_filename.c_str(), std::ofstream::out | std::ofstream::app);
    basic_outdata << N << std::endl;
    basic_outdata.close();

    char str[12];
    sprintf(str, "%d", number);
    std::string frame_name(str);

    std::string positions_filename(std::string("particle_x_")+frame_name);
    FILE_IO::Write_Binary(output_directory,positions_filename,x_n);
    std::string velocities_filename(std::string("particle_v_")+frame_name);
    FILE_IO::Write_Binary(output_directory,velocities_filename,v_n);
  }

  void Read_State(const int number){
    Read_State(x_n,v_n,N,output_directory);
    x_np1.resize(N);residual.resize(N);x_hat.resize(N);
  }

  static bool Read_State(TVect& x,TVect& v,int& N,std::string output_directory,const int number){
    std::ifstream basic_indata;
    int last_frame;
    std::string simulation_data_filename(output_directory+std::string("/simulation_info.dat"));
    basic_indata.open(simulation_data_filename.c_str());
    basic_indata >> last_frame;
    basic_indata >> N;
    basic_indata.close();

    if(number>last_frame || number<0) return false;

    char str[12];
    sprintf(str, "%d", number);
    std::string frame_name(str);
    x.resize(N);v.resize(N);
    std::string positions_filename(std::string("particle_x_")+frame_name);
    FILE_IO::Read_Binary(output_directory,positions_filename,x);
    std::string velocities_filename(std::string("particle_v_")+frame_name);
    FILE_IO::Read_Binary(output_directory,velocities_filename,v);

    return true;
  }

};
}
#endif
