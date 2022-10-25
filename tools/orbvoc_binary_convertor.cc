#include <iostream>
#include <string>

#include "cam/orb_feature/orb_vocabulary.h"

using namespace std;

using namespace ORB_SLAM_FUSION;

int main(int argc, char** argv) {
  string filename_read = string(argv[1]);
  string filename_save = string(argv[2]);
  ORBVocabulary* orb_voc = new ORBVocabulary();

  cout << "Loading ORBvoc from \".txt\"..." << endl;
  orb_voc->loadFromTextFile(filename_read);
  cout << "Done!" << endl;
  cout << "Saving ORBvoc to \".bin\"..." << endl;
  orb_voc->saveToBinaryFile(filename_save);
  cout << "Done!" << endl;
  cout << "Loading ORBvoc from \".bin\"..." << endl;
  orb_voc->loadFromBinaryFile(filename_save);
  cout << "Done!" << endl;
  cout << "Converting succeeded!" << endl;
  
  return 0;
}
