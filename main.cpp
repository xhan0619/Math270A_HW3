#include <cmath>
#include "Tools.h"
#include "ImplicitQRSVD.h"
#include "SymmetricTridiagonal.h"
#include "SimulationDriver.h"
#include "EnergyTests.h"

void EnergyTest(){
  typedef double T;
  typedef Eigen::Matrix<T,Eigen::Dynamic,1> TVect;
  int N=5;
  T a=(T)0,b=(T)1;
  T dX=(b-a)/(T)(N-1);
  JIXIE::NeoHookean<T> nh((T)1);
  JIXIE::LinearElasticity<T> le((T)1);
  JIXIE::FEMHyperelasticity<T> fem(a,dX,N,nh);
  TVect x(N);
  for(int i=0;i<N;i++) x(i)=(T).7*(a+dX*(T)i);

  JIXIE::EnergyTest<T> et("output",fem,10);
  et.RefinementTest(x);

}

void ElasticitySimulation(){


  typedef double T;
  typedef Eigen::Matrix<T,Eigen::Dynamic,1> TVect;

  JIXIE::ElasticityParameters<T> parameters;
  parameters.N=20;
  parameters.a=(T)0;
  T b=(T)1;
  parameters.dX=(b-parameters.a)/(T)(parameters.N-1);
  parameters.dt=(T).01;
  parameters.output_dir=std::string("output");
  parameters.rho=(T)1;
  parameters.k=(T)1;
  parameters.Newton_tol=(T)1e-8;
  parameters.max_newton_it=40;
  parameters.final_time=(T)12;
  parameters.frames_per_second=120;
  parameters.tb = (T)-1; // XH initialize tb
  JIXIE::ElasticityDriver<T> driver(parameters);
  bool verbose=true;
  driver.RunSimulation(verbose);
}

void ConvertBinaryToDat(){
  typedef double T;
  typedef Eigen::Matrix<T,Eigen::Dynamic,1> TVect;
  TVect x,v; int N=0,frame=0;

  std::string data_dir("output");
  std::string output_dat_dir("output/matlab");

  while(JIXIE::ElasticityDriver<T>::Read_State(x,v,N,data_dir,frame)){
    char str[12];
    sprintf(str, "%d", frame++);
    std::string frame_name(str);
    std::string positions_string(std::string("particle_x_")+frame_name);
    std::string velocities_string(std::string("particle_v_")+frame_name);
    FILE_IO::Write_DAT_File(std::string(output_dat_dir+std::string("/")+positions_string+std::string(".dat")),x);
    FILE_IO::Write_DAT_File(std::string(output_dat_dir+std::string("/")+velocities_string+std::string(".dat")),x);
  }
}

int main()
{
  //EnergyTest();
  ElasticitySimulation();
  ConvertBinaryToDat();
}
