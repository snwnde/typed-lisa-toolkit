{{ objname | escape | underline}}

.. currentmodule:: {{ module }}

{% if objtype != 'class' %}
.. auto{{ objtype }}:: {{ objname }}
{% endif %}

{% if objtype == 'class' %}
.. autoclass:: {{ objname }}
   :members:
   :member-order: groupwise
   :inherited-members:
{% endif %}