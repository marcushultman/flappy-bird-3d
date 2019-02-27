#pragma once

#include <functional>
#include <memory>
#include <string>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/mat4x4.hpp>
#include <glm/vec4.hpp>

namespace flappy {

struct Shader {
  GLuint vertex;
  GLuint fragment;
  GLuint program;
};

// class Game {
//  public:
//   void jump() {}
//   void restart() {}
//   void draw() {

//   }

//   // Bird _bird;
// };

class Game {
 public:
  using OnExit = std::function<void()>;

  explicit Game(const OnExit &on_exit) : _on_exit{on_exit} {
    // glm::mat4 matrix;
    // glm::vec4 vec;
    // auto test = matrix * vec;
  }

  void onKey(int key, int scancode, int action, int mods) {
    if (action == GLFW_PRESS) {
      return;
    }
    switch (key) {
      case GLFW_KEY_ESCAPE:
        return _on_exit();
        // case GLFW_KEY_SPACE: return _game->jump();
        // case GLFW_KEY_R: return _game->restart();
    }
  }

  void draw() {
    // _game->draw();
  }

 private:
  std::function<void()> _on_exit;
};

}  // namespace flappy
