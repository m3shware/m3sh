{{ fullname | escape | underline }}

.. automodule:: {{ fullname }}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: Attributes

   .. autosummary::
   {% for item in attributes %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% set mixins = [] %}
   {% set concrete = [] %}

   {% for item in classes %}
   {% if 'Mixin' in item %}
      {% set _ = mixins.append(item) %}
   {% else %}
      {% set _ = concrete.append(item) %}
   {% endif %}
   {% endfor %}

   {% block concrete %}
   {% if concrete %}
   .. rubric:: Classes

   .. autosummary::
      :toctree:
      :template: class-toc-vtk.rst
   {% for item in concrete %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block mixins %}
   {% if mixins %}
   .. rubric:: Mixins

   .. autosummary::
      :toctree:
      :template: class-toc-vtk.rst
   {% for item in mixins %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block exceptions %}
   {% if exceptions %}
   .. rubric:: Exceptions

   .. autosummary::
      :toctree:
      :template: exception.rst
   {% for item in exceptions %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}
   
   {% block functions %}
   {% if functions %}
   .. rubric:: Functions

   .. autosummary::
      :toctree:
      :template: function.rst
   {% for item in functions %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}
