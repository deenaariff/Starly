var app = angular.module('Sentiment',[]);

app.controller("MainCtrl", ['$scope', '$http', function($scope, $http) {

  $scope.sentence = "";
  $scope.width = [0,0,0]
  $scope.score = ["","",""]
  $scope.total_width = 200;
  $scope.stars = 0;
  $scope.test = "TEST"

  var lock = [true,true,true];
  var base = window.location.href;

  $scope.predict = function () {
    console.log("Prediction")
    if($scope.sentence != "") {
      (function (index) {
        $http.get(base+$scope.sentence, {}).then(function(data) {
          console.log(data.data)
          move(data.data, index);
        }, function(err) {
          console.log(err)
        });
      })(0);
    } else {
      $scope.stars = 0;
      for(var i = 0; i <= 6; i++) {
        $scope.score[0] = ""    
        $scope.width[0] = 0 + 'px';
      }
    }
  }

  $scope.predict2 = function () {
    for (var i = 1; i <= 2; i++) {
      if($scope.sentence != "") {
        (function (index) {
          $http.get(base+"predict/algo"+i+"/"+$scope.sentence, {}).then(function(data) {
            move(data.data, index);
          });
        })(i);
      } else {
        $scope.score[i] = ""
        $scope.stars = 0;
        $scope.width[i] = 0 + 'px';
      }
    }
  }

  var move = function (score, index) {
    if(index == 0) score = (score/2.0)+0.5;
    if(index == 0) {
      $scope.stars = Math.ceil(score*5);
    }
    $scope.score[index] = Math.floor(score*100);
    $scope.width[index] = Math.floor(score*$scope.total_width) + 'px';
  }


}]);