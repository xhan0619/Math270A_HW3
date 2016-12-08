#ifndef _SYMMETRIC_TRIDIAGONAL_
#define _SYMMETRIC_TRIDIAGONAL_

#include <Eigen/Dense>
#include <Eigen/Core>
#include "ImplicitQRSVD.h"

namespace JIXIE {

/**
    Symmetric tridiagonal matrix class.
*/
template <class T>
class SymmetricTridiagonal {

typedef Eigen::Matrix<T, Eigen::Dynamic, 1> TVect;
typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> TMat;
TVect alpha,beta;
int n;

//LDLT
TVect d,mu;

//QR
std::vector<GivensRotation<T> > Q;//the ith entry in the array introduces a zero in the i+1,i entry
TVect r1,r2,r3;

//GMRES
std::vector<TVect> q;
std::vector<TVect> h; // hold columns of H;
TMat H; // hold actual H
std::vector<GivensRotation<T> > QQ; // Might not needed


public:
  SymmetricTridiagonal(const int n_input):n(n_input){alpha.resize(n_input);beta.resize(n_input-1);
    alpha=TVect::Zero(n);
    beta=TVect::Zero(n-1);
  }
  SymmetricTridiagonal(const TMat& A):n(A.rows()){
    assert(A.rows()==A.cols());
    alpha.resize(A.rows());
    beta.resize(A.rows()-1);
    *this=A;
  }

  void SetToZero(){
    alpha(0)=(T)0;
    for(int i=1;i<n;i++){
      alpha(i)=(T)0;
      beta(i-1)=(T)0;}
  }

  void Multiply(const TVect& x,TVect& result)const {
    if(result.size()!=n || n<2 || x.size()!=n){
      std::cout<<"SymmetricTridiagonal of inappropriate size" << std::endl;
      exit(1);}

    result(0)=alpha(0)*x(0)+beta(0)*x(1);
    for(int i=1;i<n-1;i++) result(i)=beta(i-1)*x(i-1)+alpha(i)*x(i)+beta(i)*x(i+1);
    result(n-1)=beta(n-2)*x(n-2)+alpha(n-1)*x(n-1);
  }

  T& operator()(const int i,const int j){
    if(abs(i-j)>1 || i<0 || i>n-1 || j<0 || j>n-1){
      exit(1);
      return alpha(0);}
    else if(i==j)
      return alpha(i);
    else
      return beta(std::min(i,j));
  }

  void operator=(const TMat& A){
    assert(A.rows()==n && A.cols()==n);
    alpha(0)=A(0,0);
    for(int i=1;i<n;i++){
      alpha(i)=A(i,i);
      beta(i-1)=A(i-1,i);}
  }

  void Set(TMat& A)const{
    A.resize(n,n);
    A=TMat::Zero(n,n);
    A(0,0)=alpha(0);
    A(0,1)=beta(0);
    A(1,0)=beta(0);
    for(int i=1;i<n-1;i++){
      A(i,i)=alpha(i);
      A(i,i+1)=beta(i);
      A(i+1,i)=beta(i);
    }
    A(n-1,n-1)=alpha(n-1);
  }

  void LDLT(){
    d.resize(n);
    mu.resize(n-1);

    d(0)=alpha(0);
    for(int i=1;i<n;i++){
      mu(i-1)=beta(i-1)/d(i-1);
      d(i)=alpha(i)-mu(i-1)*mu(i-1)*d(i-1);
    }
  }

  void Set_L(TMat& L){
    if(d.size()!=n || mu.size()!=n-1) return;//LDLT is stale
    L.resize(n,n);
    L=TMat::Zero(n,n);
    L(0,0)=(T)1;
    for(int i=1;i<n;i++){
      L(i,i-1)=mu(i-1);
      L(i,i)=(T)1;
    }
  }

  void Set_D(TMat& D)const{
    if(d.size()!=n || mu.size()!=n-1) return;//LDLT is stale
    D.resize(n,n);
    D=TMat::Zero(n,n);
    for(int i=0;i<n;i++)
      D(i,i)=d(i);
  }

  void Set_R(TMat& R)const{
    R.resize(n,n);
    R=TMat::Zero(n,n);
    for(int i=0;i<n-2;i++){
      R(i,i)=r1(i);R(i,i+1)=r2(i);R(i,i+2)=r3(i);
    }
    R(n-2,n-2)=r1(n-2);R(n-2,n-1)=r2(n-2);
    R(n-1,n-1)=r1(n-1);
  }

  void Set_Q(TMat& Qm)const{
    Qm.resize(n,n);
    Qm=TMat::Identity(n,n);
    for(int i=n-2;i>=0;i--)
      Q[i].transposeRowRotation(Qm);
  }

  void QRowRotation(TVect& x){
    for(int i=n-2;i>=0;i--)
      Q[i].transposeRowRotation(x);
  }

  void QTransposeRowRotation(TVect& x){
    for(int i=0;i<n-1;i++)
      Q[i].rowRotation(x);
  }

  void QRSolve(TVect& x,const TVect& b){
    QR();
    TVect QTb;
    QTb.resize(n);
    QTb=b;
    QTransposeRowRotation(QTb);
    x(n-1)=QTb(n-1)/r1(n-1);
    x(n-2)=(QTb(n-2)-r2(n-2)*x(n-1))/r1(n-2);
    for(int i=n-3;i>=0;i--)
      x(i)=(QTb(i)-r2(i)*x(i+1)-r3(i)*x(i+2))/r1(i);
  }

  void QR(){
    typedef Eigen::Matrix<T,2,1> TV2;

    assert(n>2);
    Q.resize(n-1);
    r1.resize(n);
    r2.resize(n-1);
    r3.resize(n-2);

    T alpha_bar=alpha(0);
    T beta_bar=beta(0);

    for(int j=0;j<n-2;j++){
      //compute Givens that defines rjj,rj+1,rjj+2
      Q[j].SetIK(j,j+1);
      Q[j].compute(alpha_bar,beta(j));
      r1(j)=std::sqrt(alpha_bar*alpha_bar+beta(j)*beta(j));
      //compute rjj+1 and alpha bar
      TV2 temp(beta(j),alpha(j+1));
      Q[j].rowRotation(temp);
      r2(j)=temp(0);
      alpha_bar=temp(1);
      //compute rjj+2 and beta_bar
      temp(0)=(T)0;temp(1)=beta(j+1);
      Q[j].rowRotation(temp);
      r3(j)=temp(0);
      beta_bar=temp(1);
    }
    //the last givens only effects two columns of r
    //compute Givens that defines rjj,rj+1,rjj+2
    Q[n-2].SetIK(n-2,n-1);
    Q[n-2].compute(alpha_bar,beta(n-2));
    r1(n-2)=std::sqrt(alpha_bar*alpha_bar+beta(n-2)*beta(n-2));
    //compute rjj+1 and alpha bar
    TV2 temp(beta_bar,alpha(n-1));
    Q[n-2].rowRotation(temp);
    r2(n-2)=temp(0);
    r1(n-1)=temp(1);
  }

  void GMRESSolve(TVect& x,const TVect& b){
      assert(n>2); // Can't GMRESSole if matrix is too small.
      // initialize all list to empty
      QQ.resize(0);
      h.resize(0);
      q.resize(0);

      GivensRotation<T> r; // GivensRotation holder
      // T residualBound = 0.00000000000001; //TODO: find the right bound
      const ScalarType<T> residualBound = 128 * std::numeric_limits<ScalarType<T> >::epsilon();
      T residual = 1;  // Set intial residual large enough
      TVect t;         // t is the residual vector
      t.resize(2);  t(0) = b.norm(); t(1) = 0; //Initially it has size 2

      // Generate Arnoldi Basis following Algorithm 33.1
      q.push_back(b/b.norm());
      int numOfBasis = 1;
      TVect h_temp; // hold new h_column
      while (numOfBasis <= n && std::abs(residual) > residualBound){
        TVect v; v.resize(n);
        this->Multiply(q[numOfBasis-1], v); // v = Aq_k
        h_temp.resize(numOfBasis+1); // h has numOfBasis + 1 nonzero entries
                                    // (think about the first step)

        for (int i = 0; i < numOfBasis; i++){
            h_temp(i) = q[i].dot(v);
            v -= h_temp(i)*q[i];
        }
        h_temp(numOfBasis) = v.norm(); 
        if (h_temp(numOfBasis) != (T)0)
            q.push_back(v / h_temp(numOfBasis));
        else 
            q.push_back(v/v.norm()); // if v is in the previous Krylov space, so will the solution x,
                                     // hence what the new q is doesn't matter
        numOfBasis++; // At this point, we added a new basis

        // Givens rotate h_temp with all the previously stored givens 
        for (int i = 0; i < QQ.size(); i++){
          QQ[i].rowRotation(h_temp);
        }
        
        // now add a new givens and check the residual
        // thinking about the first givens helps you get the indices right
        // The first givens affect the 1st and 2nd row and flips h(2,1) to h(1,1)
        GivensRotation<T> r(h_temp(numOfBasis-2), h_temp(numOfBasis-1), numOfBasis-2, numOfBasis-1);
        QQ.push_back(r); // push the new givens to list
        r.rowRotation(h_temp);
        h.push_back(h_temp); // now move the givens rotated h to the list 
        r.rowRotation(t); // rotate the get residual
        residual = t(numOfBasis-1); // residual is the last entry in t.
        t.conservativeResize(numOfBasis+1); //resize t
        t(numOfBasis) = (T)0; // make sure to initialize this
      }

      // now form H
      H = TMat::Zero(numOfBasis, numOfBasis-1);
      for (int j = 0; j < numOfBasis-1; j++){
        for (int i = 0; i < j+2; i++){
          H(i,j) = h[j](i); // fill in H (index is a bitch)
        }
      }

      // back substitution to find lambda (using the notation from lecture notes 11/04)
      TVect lambda;
      lambda.resize(numOfBasis-1);
      int k = numOfBasis-1;
      lambda(k-1) = t(k-1)/H(k-1, k-1);
      for (int i = k-2; i >=0; i--){
        T temp = t(i);
        for (int j = k-1; j>i; j--){
          temp -= H(i,j) * lambda(j);
        }
        lambda(i) = temp/H(i,i);
      }

      // now form x
      x = TVect::Zero(n);
      for (int i = 0; i< numOfBasis-1; i++){ // the last basis will not be a part of the sol
        x += lambda(i) * q[i];
      }
  }
};
}
#endif
