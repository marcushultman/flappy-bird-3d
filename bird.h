namespace flappy {

struct VBO {
  GLuint positions;
  GLuint colors;
};

const GLfloat diamond[4][2] = {
{  0.0,  1.0  }, /* Top point */
{  1.0,  0.0  }, /* Right point */
{  0.0, -1.0  }, /* Bottom point */
{ -1.0,  0.0  } }; /* Left point */

const GLfloat colors[4][3] = {
{  1.0,  0.0,  0.0  }, /* Red */
{  0.0,  1.0,  0.0  }, /* Green */
{  0.0,  0.0,  1.0  }, /* Blue */
{  1.0,  1.0,  1.0  } }; /* White */

constexpr kVertSrc = R"s(
#version 150
// in_Position was bound to attribute index 0 and in_Color was bound to attribute index 1
in  vec2 in_Position;
in  vec3 in_Color;

// We output the ex_Color variable to the next shader in the chain
out vec3 ex_Color;
void main(void) {
    // Since we are using flat lines, our input only had two points: x and y.
    // Set the Z coordinate to 0 and W coordinate to 1

    gl_Position = vec4(in_Position.x, in_Position.y, 0.0, 1.0);

    // GLSL allows shorthand use of vectors too, the following is also valid:
    // gl_Position = vec4(in_Position, 0.0, 1.0);
    // We're simply passing the color through unmodified

    ex_Color = in_Color;
}
)s";

constexpr kFragSrc = R"s(
#version 150
// It was expressed that some drivers required this next line to function properly
precision highp float;

in  vec3 ex_Color;
out vec4 gl_FragColor;

void main(void) {
    // Pass through our original color with full opacity.
    gl_FragColor = vec4(ex_Color,1.0);
}
)s";

class Bird {
 public:
  Bird() {
    _shader = {glCreateShader(GL_VERTEX_SHADER),
               glCreateShader(GL_FRAGMENT_SHADER),
               glCreateProgram()};

    glGenVertexArrays(1, &_vao);
    glBindVertexArray(_vao);

    glGenBuffers(2, &_vbo);

    glBindBuffer(GL_ARRAY_BUFFER, _vbo.positions);
    glBufferData(GL_ARRAY_BUFFER, 8 * sizeof(GLfloat), diamond, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, _vbo.colors);
    glBufferData(GL_ARRAY_BUFFER, 12 * sizeof(GLfloat), colors, GL_STATIC_DRAW);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(1);


    glShaderSource(_shader.fragment, 1, static_cast<const GLchar**>(&kFragSrc), 0);
    glShaderSource(_shader.vertex, 1, static_cast<const GLchar**>(&kVertSrc), 0);
    glCompileShader(_shader.program);


    glShaderSource(vertexshader, 1, (const GLchar**)&vertexsource, 0);
    glCompileShader(vertexshader);


  }

  void draw() {

  }

  GLuint _vao;
  VBO _vbo;
  Shader _shader;
};

}
