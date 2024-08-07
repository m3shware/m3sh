{{ name | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :show-inheritance:
   :no-inherited-members:
   
   {%- block attributes %}
   {%- if attributes %}
   {%- set glob = namespace(skip=False) %}
   {%- for item in attributes %}
   {%- if not glob.skip %}
   {% if item not in inherited_members %} 
   .. rubric:: Attributes

   {%- set glob.skip = True %}
   {%- endif %}
   {%- endif %}
   {%- endfor %}

   {% if glob.skip -%}
   .. autosummary::
      :toctree:
      :template: attribute.rst
   {%- endif %}
   {% for item in attributes -%}
   {%- if item not in inherited_members %}
      ~{{ name }}.{{ item }}
   {%- endif %}
   {%- endfor %}
   {%- endif %}
   {%- endblock %}

   {%- block methods %}
   {% if methods %}
   .. rubric:: Methods

   .. autosummary::
      :toctree: 
      :template: function.rst
   {% for item in methods -%}
   {%- if item not in inherited_members %}
      ~{{ name }}.{{ item }}
   {%- endif %}
   {%- endfor %}
   {%- endif %}
   {%- endblock %}
